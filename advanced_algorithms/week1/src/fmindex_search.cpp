#include <sstream>

#include <filesystem>
#include <benchmark/benchmark.h>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/io/sequence_file/all.hpp>
#include <seqan3/search/fm_index/all.hpp>
#include <seqan3/search/search.hpp>

//using Index = decltype(seqan3::fm_index{std::vector<std::vector<seqan3::dna5>>{}}); // Some hack
seqan3::fm_index<seqan3::dna5,seqan3::text_layout::collection>  fm_index;// construct fm-index
std::vector<std::vector<seqan3::dna5>> queries40;
std::vector<std::vector<seqan3::dna5>> queries60;
std::vector<std::vector<seqan3::dna5>> queries80;
std::vector<std::vector<seqan3::dna5>> queries100;

//// Copy and paste... jey!

///// Functions to benchmarks /////

void search_fm_index40() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{0}};
    auto search_results = seqan3::search(queries40, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index60() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{0}};
    auto search_results = seqan3::search(queries60, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index80() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{0}};
    auto search_results = seqan3::search(queries80, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index100() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{0}};
    auto search_results = seqan3::search(queries100, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_1_40() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{1}};
    auto search_results = seqan3::search(queries40, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_1_60() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{1}};
    auto search_results = seqan3::search(queries60, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_1_80() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{1}};
    auto search_results = seqan3::search(queries80, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_1_100() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{1}};
    auto search_results = seqan3::search(queries100, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_2_40() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{2}};
    auto search_results = seqan3::search(queries40, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_2_60() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{2}};
    auto search_results = seqan3::search(queries60, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_2_80() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{2}};
    auto search_results = seqan3::search(queries80, fm_index, cfg);
    for (auto hit : search_results);
}

void search_fm_index_error_2_100() {
    seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{2}};
    auto search_results = seqan3::search(queries100, fm_index, cfg);
    for (auto hit : search_results);
}

///// BENCHMARKS FUNCTIONS /////

static void BM_search_fm_index_no_error_40(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index40();
}

static void BM_search_fm_index_no_error_60(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index60();
}

static void BM_search_fm_index_no_error_80(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index80();
}

static void BM_search_fm_index_no_error_100(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index100();
}

static void BM_search_fm_index_error_1_40(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_1_40();
}

static void BM_search_fm_index_error_1_60(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_1_60();
}

static void BM_search_fm_index_error_1_80(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_1_80();
}

static void BM_search_fm_index_error_1_100(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_1_100();
}

static void BM_search_fm_index_error_2_40(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_2_40();
}

static void BM_search_fm_index_error_2_60(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_2_60();
}

static void BM_search_fm_index_error_2_80(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_2_80();
}

static void BM_search_fm_index_error_2_100(benchmark::State& state) {

    for (auto _ : state)
        search_fm_index_error_2_100();
}

///// CALL BENCHMARKS /////

//// Exercise 6
BENCHMARK(BM_search_fm_index_no_error_40);
BENCHMARK(BM_search_fm_index_no_error_60);
BENCHMARK(BM_search_fm_index_no_error_80);
BENCHMARK(BM_search_fm_index_no_error_100);
BENCHMARK(BM_search_fm_index_error_1_40);
BENCHMARK(BM_search_fm_index_error_1_60);
BENCHMARK(BM_search_fm_index_error_1_80);
BENCHMARK(BM_search_fm_index_error_1_100);
BENCHMARK(BM_search_fm_index_error_2_40);
BENCHMARK(BM_search_fm_index_error_2_60);
BENCHMARK(BM_search_fm_index_error_2_80);
BENCHMARK(BM_search_fm_index_error_2_100);

int main(int argc, char const* const* argv) {
    seqan3::argument_parser parser{"fmindex_search", argc, argv, seqan3::update_notifications::off};

    parser.info.author = "SeqAn-Team";
    parser.info.version = "1.0.0";

    size_t nrOfQueries{0};
    parser.add_option(nrOfQueries, 'n', "number-of-queries", "Number of queries");

    std::string output_file;
    parser.add_option(output_file, '\0', "out", "path to output file");

    size_t iterations{1};
    parser.add_option(iterations, 'i', "iterations", "Number of iterations");

    std::string selection{};
    parser.add_option(selection, '\0', "selection", "Regex to select benchmarks");

    auto index_path = std::filesystem::path{};
    parser.add_option(index_path, '\0', "index", "path to the query file");

    auto query_file_40 = std::filesystem::path{};
    parser.add_option(query_file_40, '\0', "40", "path to the query file");

    auto query_file_60 = std::filesystem::path{};
    parser.add_option(query_file_60, '\0', "60", "path to the query file");

    auto query_file_80 = std::filesystem::path{};
    parser.add_option(query_file_80, '\0', "80", "path to the query file");

    auto query_file_100 = std::filesystem::path{};
    parser.add_option(query_file_100, '\0', "100", "path to the query file");

    try {
         parser.parse();
    } catch (seqan3::argument_parser_error const& ext) {
        seqan3::debug_stream << "Parsing error. " << ext.what() << "\n";
        return EXIT_FAILURE;
    }

    // loading our files
    auto query_stream40     = seqan3::sequence_file_input{query_file_40};

    size_t i{0};

    // read query into memory
    for (auto& record : query_stream40) {
        if (i == nrOfQueries) break; // bad but other resize version leads to a segfault
        ++i;
        queries40.push_back(record.sequence());
    }

    auto query_stream_60     = seqan3::sequence_file_input{query_file_60};

    i = 0;

    // read query into memory
    for (auto& record : query_stream_60) {
        if (i == nrOfQueries) break; // bad but other resize version leads to a segfault
        queries60.push_back(record.sequence());
        ++i;
    }

    auto query_stream_80     = seqan3::sequence_file_input{query_file_80};

    i = 0;

    // read query into memory
    for (auto& record : query_stream_80) {
        if (i == nrOfQueries) break; // bad but other resize version leads to a segfault
        queries80.push_back(record.sequence());
        ++i;
    }

    auto query_stream_100     = seqan3::sequence_file_input{query_file_100};

    i = 0;

    // read query into memory
    for (auto& record : query_stream_100) {
        if (i == nrOfQueries) break; // bad but other resize version leads to a segfault
        queries100.push_back(record.sequence());
        ++i;
    }

    // loading fm-index into memory
    {
        seqan3::debug_stream << "Loading 2FM-Index ... " << std::flush;
        std::ifstream is{index_path, std::ios::binary};
        cereal::BinaryInputArchive iarchive{is};
        iarchive(fm_index);
        seqan3::debug_stream << "done\n";
    }

    std::string out{"--benchmark_out="};

    out += output_file;

    std::string repetitions{"--benchmark_repetitions="};

    repetitions += std::to_string(iterations);

    char* benchmark_out{(char*)out.c_str()};
    char* benchmark_out_format{(char*)"--benchmark_out_format=csv"};
    char* benchmark_repetitions{(char*)repetitions.c_str()};

    char* benchmark_filter{};

    if (!selection.empty()) {
        // run selected benchmarks
        std::string filter{"--benchmark_filter="};

        filter += selection;

        benchmark_filter = {(char*)filter.c_str()};
        int argc_BM =5;
        char* argv_BM[5];
        argv_BM[4] = benchmark_filter;

        argv_BM[1] = benchmark_out;
        argv_BM[2] = benchmark_out_format;
        argv_BM[3] = benchmark_repetitions;

        ::benchmark::Initialize(&argc_BM, argv_BM);
        if (::benchmark::ReportUnrecognizedArguments(argc_BM, argv_BM)) return 1;
        ::benchmark::RunSpecifiedBenchmarks();
        return 0;
    } else {
        std::cout << "==CHECKT\n";

        // run all benchmarks
        int argc_BM = 4;
        char *argv_BM[4];

        argv_BM[1] = benchmark_out;
        argv_BM[2] = benchmark_out_format;
        argv_BM[3] = benchmark_repetitions;

        ::benchmark::Initialize(&argc_BM, argv_BM);
        if (::benchmark::ReportUnrecognizedArguments(argc_BM, argv_BM)) return 1;
        ::benchmark::RunSpecifiedBenchmarks();
        return 0;
    }
}
