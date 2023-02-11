#include <sstream>

#include <filesystem>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/io/sequence_file/all.hpp>
#include <seqan3/search/fm_index/fm_index.hpp>
#include <seqan3/search/search.hpp>

bool printResults{false};
bool saveResults{false};
bool runBenchmark{false};

size_t numberOfRuns{1};

void benchmarkFindOccurrences(std::vector<seqan3::dna5> const& ref, std::vector<seqan3::dna5> const& query) {

    std::vector<size_t> positions{};

    auto position = ref.begin();

    while (true) {
        position = std::search(position, ref.end(), query.begin(), query.end());

        if (position != ref.end()) {
            positions.push_back(std::distance(ref.begin(), position));
        } else {
            break;
        }

        ++position;
    }

}

// prints out all occurences of query inside of ref
void findOccurrences(std::vector<seqan3::dna5> const& ref, std::vector<seqan3::dna5> const& query) {

    if (!printResults && !saveResults) {
        seqan3::debug_stream << "Find Occurrences: the results of the search will nighter saved nor printed!\n";
    }

    //const std::boyer_moore_searcher searcher(query.begin(), query.end());

    auto position = ref.begin();

    std::vector<size_t> positions{};

    while (true) {
        position = std::search(position, ref.end(), query.begin(), query.end());

        if (position != ref.end()) {
            positions.push_back(std::distance(ref.begin(), position));
        } else {
            break;
        }

        ++position;
    }

    if (printResults) {
        // Print
    }

    if (saveResults) {
        // Save
    }

    seqan3::debug_stream << "The query occurs " << positions.size() << " times in the reference sequence";
}

int main(int argc, char const* const* argv) {
    seqan3::argument_parser parser{"naive_search", argc, argv, seqan3::update_notifications::off};

    parser.info.author = "SeqAn-Team";
    parser.info.version = "1.0.0";

    auto reference_file = std::filesystem::path{};
    parser.add_option(reference_file, '\0', "reference", "path to the reference file");

    auto query_file = std::filesystem::path{};
    parser.add_option(query_file, '\0', "query", "path to the query file");

    auto output_file = std::filesystem::path{};
    parser.add_option(output_file, '\0', "out", "path to out put file");

    parser.add_option(numberOfRuns, 'i', "iterations", "Number of iterations");

    parser.add_flag(runBenchmark, 'b', "benchmark", "Set to run in Benchmark mode");
    parser.add_flag(printResults, 'p', "print", "Set to print results to commandline");

    try {
         parser.parse();
    } catch (seqan3::argument_parser_error const& ext) {
        seqan3::debug_stream << "Parsing error. " << ext.what() << "\n";
        return EXIT_FAILURE;
    }

    // loading our files
    auto reference_stream = seqan3::sequence_file_input{reference_file};
    auto query_stream     = seqan3::sequence_file_input{query_file};

    // read reference into memory
    std::vector<std::vector<seqan3::dna5>> reference;
    for (auto& record : reference_stream) {
        reference.push_back(record.sequence());
    }

    // read query into memory
    std::vector<std::vector<seqan3::dna5>> queries;
    for (auto& record : query_stream) {
        queries.push_back(record.sequence());
    }

    //!TODO !CHANGEME here adjust the number of searches

    if (!runBenchmark) queries.resize(10); // will reduce the amount of searches

    seqan3::debug_stream << "Starting search!\n";
    //! search for all occurences of queries inside of reference

    if (runBenchmark) {

        queries.resize(10);

        seqan3::debug_stream << "Benchmark find Occurrences:";
        std::ofstream ofstream(output_file);


        for(size_t i = 0; i < numberOfRuns; ++i) {

            auto start = std::chrono::system_clock::now();

            for (auto &r: reference) {
                for (auto &q: queries) {
                    benchmarkFindOccurrences(r, q);
                }
            }

            auto end = std::chrono::system_clock::now();

            std::chrono::duration<double> elapsed = end - start;

            ofstream << elapsed.count() << "\n";


            std::cout << "Simple binary search required " << elapsed.count() << " seconds to run\n";

            seqan3::debug_stream << "Naive searched needed " << elapsed.count() << " seconds to find all occurrences";
        }
        ofstream.close();

    }   else {
        for (auto &r: reference) {
            for (auto &q: queries) {
                findOccurrences(r, q);
            }
        }
    }

    return 0;
}
