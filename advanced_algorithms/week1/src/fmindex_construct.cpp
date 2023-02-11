#include <sstream>

#include <filesystem>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/io/sequence_file/all.hpp>
#include <seqan3/search/fm_index/fm_index.hpp>
#include <seqan3/search/search.hpp>

int main(int argc, char const* const* argv) {
    seqan3::argument_parser parser{"fmindex_construct", argc, argv, seqan3::update_notifications::off};

    std::cout << "Running FM-Index construct!\n";

    parser.info.author = "SeqAn-Team";
    parser.info.version = "1.0.0";

    auto reference_file = std::filesystem::path{};
    parser.add_option(reference_file, '\0', "reference", "path to the reference file");

    auto index_path = std::filesystem::path{};
    parser.add_option(index_path, '\0', "index", "path to the query file");

    try {
         parser.parse();
    } catch (seqan3::argument_parser_error const& ext) {
        seqan3::debug_stream << "Parsing error. " << ext.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Load Data!\n";

    // loading our files
    auto reference_stream = seqan3::sequence_file_input{reference_file};

    // read reference into memory
    std::vector<std::vector<seqan3::dna5>> reference;
    for (auto& record : reference_stream) {
        reference.push_back(record.sequence());
    }

    std::cout << "Constructing FM-Index!\n";

    auto start = std::chrono::system_clock::now();

    // Our index is of type `Index`
    seqan3::fm_index index{reference}; // construct fm-index

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Build Index in " << elapsed.count() << "\n";

    // saving the fmindex to storage
    {
        seqan3::debug_stream << "Saving 2FM-Index ... " << std::flush;
        std::ofstream os{index_path, std::ios::binary};
        cereal::BinaryOutputArchive oarchive{os};
        oarchive(index);
        seqan3::debug_stream << "done\n";
    }

    return 0;
}
