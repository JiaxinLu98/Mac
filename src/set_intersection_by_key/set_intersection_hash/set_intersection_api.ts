import { readSize } from '../../utils';
import { GPUSetIntersection } from './set_intersection';
import TimestampQueryManager from '../../TimestampQueryManager';

export async function setIntersectionByKey(
  device: GPUDevice,
  aData: Uint32Array,
  bData: Uint32Array,
  iters: number
) {
  const a_in = new Uint32Array(aData);
  const b_in = new Uint32Array(bData);

  const QUERIES_PER_ITER = 12; 
  const timestampManager = new TimestampQueryManager(device, iters * QUERIES_PER_ITER);
  const setIntersection = new GPUSetIntersection(device, timestampManager); 

  const aTupleCount = a_in.length / 2;

  const bufferA = device.createBuffer({
    label: 'A array buffer',
    size:   a_in.byteLength,
    usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(bufferA, 0, a_in);

  const bufferB = device.createBuffer({
    label: 'B array buffer',
    size:   b_in.byteLength,
    usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(bufferB, 0, b_in);

  const bufferBSize = device.createBuffer({
    label: 'B size uniform buffer',
    size: Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bufferBSize, 0, new Uint32Array([b_in.length]));

  const maxResultSize = Math.min(aTupleCount, b_in.length);
  const bufferResult = device.createBuffer({
    label: 'intersection result buffer',
    size:  maxResultSize * 2 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const bufferResultSize = device.createBuffer({
    label: 'intersection result size buffer',
    size:  Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bufferResultSize, 0, new Uint32Array([0]));

  let wallClockSumMs = 0;

  for(let i = 0; i < iters; i++) {
    const iterStart = performance.now();

    setIntersection.setIterationIndex(i);

    device.queue.writeBuffer(bufferResultSize, 0, new Uint32Array([0]));

    // API CALL: BUILD HASH TABLE FOR A.
    const { bufferHashTable, bufferHashTableSize } = await setIntersection.buildHashTable(aTupleCount, bufferA);
    // API CALL: GET JOIN RESULT.
    await setIntersection.getJoinResult(bufferHashTable, bufferHashTableSize, bufferB, bufferBSize, bufferResult, bufferResultSize, b_in.length);
    // API CALL
    await setIntersection.runSortPhase(bufferResultSize);

    const iterEnd = performance.now(); 
    wallClockSumMs += (iterEnd - iterStart);

    bufferHashTable.destroy();
    bufferHashTableSize.destroy();
  }

  const wallClockAvgS = (wallClockSumMs / iters) / 1000;
  console.log(`Wall-clock over ${iters} iterations: ` + `avg = ${wallClockAvgS.toFixed(9)} s/iter`);


  // Copy the result from the GPU back to a readable buffer on the CPU.
  const commandEncoder = device.createCommandEncoder();
  
  const resultSize = await readSize(device, bufferResultSize);

  const readbackBuffer = device.createBuffer({
      size: resultSize * 2 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(bufferResult, 0, readbackBuffer, 0, resultSize * 2 * Uint32Array.BYTES_PER_ELEMENT);
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultData = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();

  const timestamps = await timestampManager.downloadTimestampResult();

  let sumBuild = 0;
  let sumJoin  = 0;
  let sumSort  = 0;

  for (let i = 0; i < iters; ++i) {
    const base = i * QUERIES_PER_ITER;

    const buildTicks = timestamps[base + 1] - timestamps[base + 0];
    const joinTicks  = timestamps[base + 3] - timestamps[base + 2];
    const sortTicks  = timestamps[base + 11] - timestamps[base + 4];

    sumBuild += buildTicks;
    sumJoin  += joinTicks;
    sumSort  += sortTicks;
  }

  const avg = {
    build: sumBuild / iters,
    join:  sumJoin  / iters,
    sort:  sumSort  / iters,
  };

  const timestampPeriod = 1e-9;

  const avgSeconds = {
    build: avg.build * timestampPeriod,
    join:  avg.join  * timestampPeriod,
    sort:  avg.sort  * timestampPeriod,
  };

  console.log(
    `Average over ${iters} iterations:`,
    `build = ${avgSeconds.build.toFixed(9)} s,`,
    `join = ${avgSeconds.join.toFixed(9)} s,`,
    `sort = ${avgSeconds.sort.toFixed(9)} s`,
    `Total time = ${(avgSeconds.build + avgSeconds.join + avgSeconds.sort).toFixed(9)} s`
  );

  // Release GPU resources to allow re-entry into the run function.
  bufferA.destroy();
  bufferB.destroy();
  bufferBSize.destroy();
  bufferResult.destroy();
  bufferResultSize.destroy();
  readbackBuffer.destroy();

  return resultData;
}
