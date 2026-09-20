const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");
const path = require("path");

module.exports = {
    entry: "./src/app.ts",
    mode: "development",
    devtool: "inline-source-map",
    output: {
        filename: "main.js",
        path: path.resolve(__dirname, "dist"),
    },
    module: {
        rules: [
            {
                test: /\.(png|jpg|jpeg)$/i,
                type: "asset/resource",
            },
            {
                // Embed your WGSL files as strings
                test: /\.wgsl$/i,
                type: "asset/source",
            },
            {
                test: /\.tsx?$/,
                use: "ts-loader",
                exclude: /node_modules/,
            }
        ]
    },
    resolve: {
        extensions: [".tsx", ".ts", ".js"],
    },
    devServer: {
        static: [
            { directory: path.resolve(__dirname, 'public'), publicPath: '/' },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: "./index.html",
        }),
        new CopyWebpackPlugin({
            patterns: [
                { from: "edge.facts", to: "edge.facts", noErrorOnMissing: true },
                // 5.4GB of triangle-counting graph data; re-enable when needed.
                // { from: "graphs", to: "graphs", noErrorOnMissing: true },
            ],
        }),
    ],
};
