#include <sstream>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/io/sequence_file/all.hpp>
#include <seqan3/search/fm_index/fm_index.hpp>
#include <seqan3/search/search.hpp>

void cutQuery(size_t nrOfErrors, const std::vector<seqan3::dna5> & query, std::vector<std::vector<seqan3::dna5>> & parts);
bool verify(size_t startPosition, size_t partNr, const std::vector<std::vector<seqan3::dna5>> & parts, size_t nrOfErrors,
            const std::vector<seqan3::dna5> &reference);
bool compareSequence(const std::vector<seqan3::dna5> & part, const std::vector<seqan3::dna5> & reference,
                     const size_t startPosition, size_t i , size_t & errorCounter, size_t allowedErrors);

template<class Number>
bool isEven(Number number) {
    return ((number % 2) == 0);
}

int main(int argc, char const* const* argv) {
    seqan3::argument_parser parser{"fmindex_pigeon_search", argc, argv, seqan3::update_notifications::off};

    parser.info.author = "SeqAn-Team";
    parser.info.version = "1.0.0";

    auto reference_file = std::filesystem::path{};
    parser.add_option(reference_file, '\0', "reference", "path to the reference file");

    size_t nrOfQueries{100};
    parser.add_option(nrOfQueries, 'n', "number-of-queries", "Number of queries");

    size_t iterations{1};
    parser.add_option(iterations, 'i', "iterations", "Number of iterations");

    auto index_path = std::filesystem::path{};
    parser.add_option(index_path, '\0', "index", "path to the query file");

    auto query_file = std::filesystem::path{};
    parser.add_option(query_file, '\0', "query", "path to the query file");

    std::string output_file;
    parser.add_option(output_file, '\0', "out", "path to output file");
    try {
         parser.parse();
    } catch (seqan3::argument_parser_error const& ext) {
        seqan3::debug_stream << "Parsing error. " << ext.what() << "\n";
        return EXIT_FAILURE;
    }

    // loading our files
    auto query_stream     = seqan3::sequence_file_input{query_file};

    std::cout << "load queries \n";
    // read query into memory
    std::vector<std::vector<seqan3::dna5>> queries;
    for (auto& record : query_stream) {
        queries.push_back(record.sequence());
    }

    std::cout << "load index \n";
    // loading fm-index into memory
    using Index = decltype(seqan3::fm_index{std::vector<std::vector<seqan3::dna5>>{}}); // Some hack
    Index index; // construct fm-index
    {
        seqan3::debug_stream << "Loading 2FM-Index ... " << std::flush;
        std::ifstream is{index_path, std::ios::binary};
        cereal::BinaryInputArchive iarchive{is};
        iarchive(index);
        seqan3::debug_stream << "done\n";
    }

    std::cout << " load reference \b" << std::flush;
    // loading our files
    auto reference_stream = seqan3::sequence_file_input{reference_file};

    // read reference into memory
    // Attention: we are concatenating all sequences into one big combined sequence
    //            this is done to simplify the implementation of suffix_arrays
    std::vector<seqan3::dna5> reference;
    for (auto &record: reference_stream) {
        auto r = record.sequence();
        reference.insert(reference.end(), r.begin(), r.end());
    }
    //!TODO here adjust the number of searches
    queries.resize(nrOfQueries); // will reduce the amount of searches

    std::ofstream ofstream(output_file);

std::cout<< "search queries \n";
    for (size_t i = 1; i <=10 ;++i) {

        auto start = std::chrono::system_clock::now();
        std::vector<size_t> hits{};


        for (size_t nrOfErrors{1}; nrOfErrors <= 1; ++nrOfErrors) {
            //!TODO !ImplementMe use the seqan3::search to find a partial error free hit, verify the rest inside the text
            // Pseudo code (might be wrong):
            // for query in queries:
            for (const std::vector<seqan3::dna5> &query: queries) {
                std::unordered_set<size_t> hitSet{};


                // Copy queries, because I don't know if or how you can search range based with seqans FM-Index
                std::vector<std::vector<seqan3::dna5>> parts{};

                //      parts[3] = cut_query(3, query);
                cutQuery(nrOfErrors, query, parts);

                auto results = seqan3::search(parts, index);

                if (nrOfErrors == 0) {
                    for (auto result: results) {
                        hitSet.insert(result.reference_begin_position());
                    }
                }

                /*for (auto part : parts) {
                    std::cout << "PART ";
                    for (auto base : part) {
                        std::cout << base.to_char();
                    }

                    std::cout << "\n";
                }*/
                for (auto result: results) {

                    auto position = result.reference_begin_position();
                    size_t partNr = result.query_id();

                    for (size_t currentNrOfErrors = 1; currentNrOfErrors <= nrOfErrors; ++currentNrOfErrors) {

                        if (verify(position, partNr, parts, nrOfErrors, reference)) {

                            size_t hitPosition{position};

                            for (size_t i = 0; i < partNr; ++i) {
                                hitPosition -= parts[i].size();
                            }

                            hitSet.insert(hitPosition);
                        }
                    }
                }
                for (auto hit: hitSet) {
                    hits.push_back(hit);
                }
            }

        }
        auto end = std::chrono::system_clock::now();

        /*seqan3::configuration cfg = seqan3::search_cfg::max_error_total{seqan3::search_cfg::error_count{2}}
                                    | seqan3::search_cfg::max_error_substitution{seqan3::search_cfg::error_count{2}}
                                    | seqan3::search_cfg::max_error_insertion{seqan3::search_cfg::error_count{0}}
                                    | seqan3::search_cfg::max_error_deletion{seqan3::search_cfg::error_count{0}};
        auto search_results = seqan3::search(queries, index, cfg);
        size_t counter{0};
        std::vector<std::pair<size_t, size_t>> hitsSEQAN{};
        for (auto hit: search_results) {
            hitsSEQAN.emplace_back(hit.reference_begin_position(), hit.query_id());
        }
        std::cout << counter << "\n";

        for (auto hit : hitsSEQAN) {
            auto result = std::find(hits.begin(),hits.end(),hit.first);
            if (result == hits.end()) {
                std::cout << hit.first << "\t" << hit.second << "\n";

                for (size_t i = hit.first; i <hit.first+40; ++i ) {
                    std::cout << reference[i].to_char() ;
                }
                std::cout << "\n";
            }
        }*/

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Search requires " << elapsed.count() << " seconds to run\n";

        ofstream << elapsed.count() << "\n";
    }




       // std::cout << "We found " << hits.size() << " but seqan found " << hitsSEQAN.size() << " hits! Juhu!\n";
    ofstream.close();



    return 0;
}

void cutQuery(size_t nrOfErrors, const std::vector<seqan3::dna5> & query, std::vector<std::vector<seqan3::dna5>> & parts) {

    for (size_t i = 0; i <= nrOfErrors; ++i) {

        std::vector<seqan3::dna5> part{};
        if (nrOfErrors == 0) {
            part.assign(query.begin(), query.end());
            parts.push_back(part);
            return;
        }

        auto left = static_cast<std::_Bit_const_iterator::difference_type>(std::floor((query.size() * i) / (nrOfErrors+1)));
        auto right = static_cast<std::_Bit_const_iterator::difference_type>(std::floor((query.size() * (i+1)) / (nrOfErrors+1)));

        part.assign(query.begin()+left, query.begin()+right);

        parts.push_back(part);
    }
}

bool verify(size_t firstStartPosition, size_t partNr, const std::vector<std::vector<seqan3::dna5>> & parts, size_t nrOfErrors,
            const std::vector<seqan3::dna5> & reference) {

    auto startPosition = static_cast<int>(firstStartPosition);

    size_t nextPartNr = partNr;
    size_t errorCounter{0};

    bool leftIsFinished{false};

    for (size_t allowedErrors = 1; allowedErrors <= nrOfErrors; ++allowedErrors) {

        if (leftIsFinished) {
            startPosition += static_cast<int>(parts[nextPartNr].size());
            ++nextPartNr;
            if (reference.size() < (startPosition + parts[nextPartNr].size())) return false;
        } else if (nextPartNr == 0) {
            startPosition = static_cast<int>(firstStartPosition + parts[partNr+1].size());
            nextPartNr = partNr + 1;
            if (reference.size() < (startPosition + parts[nextPartNr].size())) return false;
            leftIsFinished = true;
        } else {
            --nextPartNr;
            startPosition -= static_cast<int>(parts[nextPartNr].size());
            if (startPosition < 0) return false;
        }

        if (!compareSequence(parts[nextPartNr], reference, startPosition, 0, errorCounter, nrOfErrors)) return false;
    }

    return true;
}

bool compareSequence(const std::vector<seqan3::dna5> & part, const std::vector<seqan3::dna5> & reference,
                     const size_t startPosition, size_t i , size_t & errorCounter, size_t allowedErrors) {

    for (; i < part.size(); ++i) {
        if (part[i] != reference[startPosition + i]) {
            ++errorCounter;

            if (errorCounter > allowedErrors) return  false;

            /*// Insertion
            if (i == 0 && startPosition != 0) {
                if (part[i] != reference[startPosition-1]) ++errorCounter;
                compareSequence(part, reference, startPosition, (i + 1), errorCounter, allowedErrors);
            } else if (compareSequence(part, reference, startPosition-1, i, errorCounter, allowedErrors)) return true;

            // Deletion
            if (compareSequence(part, reference, startPosition+1, (i + 1), errorCounter, allowedErrors)) return true;*/
        }
    }
    return true;
}
