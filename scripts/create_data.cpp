#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

bool write_int_vector_bin(const std::string& filename,
                          const std::vector<int>& data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return false;
    }

    uint64_t len = static_cast<uint64_t>(data.size());
    ofs.write(reinterpret_cast<const char*>(&len), sizeof(uint64_t));
    ofs.write(reinterpret_cast<const char*>(data.data()),
              sizeof(int) * data.size());
    return ofs.good();
}

std::vector<int> generate_sorted_random_ints(std::size_t N,
                                             int valueRange,
                                             std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(1, valueRange);
    std::vector<int> v(N);
    for (std::size_t i = 0; i < N; ++i) {
        v[i] = dist(rng);
    }
    std::sort(v.begin(), v.end());
    return v;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " N valueRange A_out.bin B_out.bin\n";
        std::cerr << "Example: " << argv[0]
                  << " 1000000 1000000 A_1e6.bin B_1e6.bin\n";
        return 1;
    }

    std::size_t N       = std::stoull(argv[1]);
    int valueRange      = std::stoi(argv[2]);
    std::string A_file  = argv[3];
    std::string B_file  = argv[4];

    std::mt19937 rng(123456);

    std::cout << "Generating A (N=" << N << ", range=1.." << valueRange << ")\n";
    auto A = generate_sorted_random_ints(N, valueRange, rng);

    std::cout << "Generating B (N=" << N << ", range=1.." << valueRange << ")\n";
    auto B = generate_sorted_random_ints(N, valueRange, rng);

    std::cout << "Writing " << A_file << " ...\n";
    if (!write_int_vector_bin(A_file, A)) return 1;

    std::cout << "Writing " << B_file << " ...\n";
    if (!write_int_vector_bin(B_file, B)) return 1;

    std::cout << "Done.\n";
    return 0;
}
