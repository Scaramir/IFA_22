#include <divsufsort.h>
//#include "../lib/libdivsufsort/lib/utils.c"
#include <sstream>

#include <filesystem>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/io/sequence_file/all.hpp>
#include <seqan3/search/fm_index/fm_index.hpp>
#include <seqan3/search/search.hpp>

#define assertm(exp, msg) assert(((void)msg, exp))

//// TO-DO:
//// 1. Is contains in right boundary necessary or redundant with bigger?
//// 2. write seperated classes for:
////        a) simple binary search
////        b) binary search with mlr
////        c) binary search with lcp
//// 3. Write tests to a separate unity test and add further tests!
//// 4. Optimize LCP!!!!
///         a) Use array instead of tree.
///         b) Use different algorithm for computing - no recursion
///         c) Implement containsLCP for right boundary
//// 5. Copy comparison functions to a utils script
//// 6. Reorganise cmd parser and main
//// 7. Code documentation

using namespace seqan3::literals;

struct Borders {
    size_t left{};
    size_t right{};
};

bool smallerEqual(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right, const saidx_t position) {

    for (size_t i = 0; i < std::min(left.size(), right.size()-position); ++i) {
        if (left[i].to_char() == right[position+i].to_char()) {
            continue;
        } else {
            return left[i].to_char() <= right[position+i].to_char();
        }
    }
    return left.size() <= right.size()-position;
}

bool biggerEqual(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right, const saidx_t position) {
    for (size_t i = 0; i < std::min(left.size(), right.size()-position); ++i) {
        if (left[i] == right[position+i]) {
            continue;
        } else {
            return left[i] >= right[position+i];
        }
    }
    return left.size() >= right.size()-position;
}

bool smaller(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right, const saidx_t position) {
    return (!biggerEqual(left, right, position));
}

bool bigger(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right, const saidx_t position) {
    return (!smallerEqual(left, right, position));
}

bool contains(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right, const saidx_t position) {
    if (left.size() > right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        if (left[i] == right[position+i]) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

size_t leftBorder (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                   const std::vector<saidx_t> & suffixArray){

    if (smaller(query, reference, suffixArray[0])) return 0;
    if (!smallerEqual(query, reference, suffixArray[suffixArray.size()-1])) return suffixArray.size();

    // can't we just use borders.right here if we'd also initialize borders{0, ref.size()}?
    size_t left{0}; size_t right{reference.size()};

    while (right - left > 1) {
        size_t middle{static_cast<size_t>(std::ceil((left+right)/2))};

        if (smallerEqual(query, reference, suffixArray[middle])) {
            right = middle;
        } else {
            left = middle;
        }
    }

    if (smallerEqual(query,reference, suffixArray[left])) {
        return left;
    } else {
        return right;
    }
}

size_t rightBorder (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                    const std::vector<saidx_t> & suffixArray) {

    if (smaller(query, reference, suffixArray[0])) return 0;
    if (contains(query, reference, suffixArray[suffixArray.size()-1])) return suffixArray.size();

    // can't we just use borders.left if we'd also initialize borders{0, ref.size()}?
    // -> size_t left{borders.left}; size_t right{borders.right};
    size_t left{0}; size_t right{reference.size()};

    while (right - left > 1) {
        size_t middle{static_cast<size_t>(std::ceil((left+right)/2))};

        if (contains(query, reference, suffixArray[middle]) || bigger(query, reference, suffixArray[middle])) {
            left = middle;
        } else {
            right = middle;
        }
    }

    return right;
}

bool binarySearch (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                   const std::vector<saidx_t> & suffixArray, Borders & borders) {

    borders.left = leftBorder(query, reference, suffixArray);
    borders.right = rightBorder(query, reference, suffixArray);

    return borders.left < borders.right;
}

bool biggerEqualMLR(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
                    const saidx_t position, size_t & mlr) {
    for (size_t i = mlr; i < std::min(left.size(), right.size()-position); ++i) {
        if (left[i] == right[position+i]) {
            ++mlr;
            continue;
        } else {
            return left[i] >= right[position+i];
        }
    }
    return left.size() >= right.size()-position;
}

bool smallerMLR(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
                const saidx_t position, size_t & mlr) {
    return (!biggerEqualMLR(left, right, position, mlr));
}

bool smallerEqualMLR(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
                     const saidx_t position, size_t & mlr) {

    for (size_t i = mlr; i < std::min(left.size(), right.size()-position); ++i) {
        if (left[i] == right[position+i]) {
            ++mlr;
            continue;
        } else {
            return left[i] <= right[position+i];
        }
    }
    return left.size() <= right.size()-position;
}

bool biggerMLR(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
               const saidx_t position, size_t & mlr) {
    return (!smallerEqualMLR(left, right, position, mlr));
}


size_t leftBorderMLR (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                      const std::vector<saidx_t> & suffixArray){

    size_t l{0};  size_t r{0};

    if (smallerMLR(query, reference, suffixArray[0], l)) return 0;
    if (!smallerEqualMLR(query, reference, suffixArray[suffixArray.size()-1], r)) return suffixArray.size();

    size_t left{0}; size_t right{reference.size()};
    size_t mlr = std::min(l,r);

    while (right - left > 1) {
        size_t middle{static_cast<size_t>(std::ceil((left+right)/2))};

        if (smallerEqualMLR(query, reference, suffixArray[middle], mlr)) {
            r = mlr;
            mlr = std::min(l,r);
            right = middle;
        } else {
            l = mlr;
            mlr = std::min(l,r);
            left = middle;
        }
    }

    if (smallerEqual(query,reference, suffixArray[left])) {
        return left;
    } else {
        return right;
    }
}

bool containsMLR(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
                 const saidx_t position, size_t & mlr) {
    if (left.size() > right.size()) return false;
    for (size_t i = mlr; i < left.size(); ++i) {
        if (left[i] == right[position+i]) {
            ++mlr;
            continue;
        } else {
            return false;
        }
    }
    return true;
}

size_t rightBorderMLR (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                    const std::vector<saidx_t> & suffixArray) {

    size_t l{0};  size_t r{0};

    if (smallerMLR(query, reference, suffixArray[0], l)) return 0;
    if (contains(query, reference, suffixArray[suffixArray.size()-1])) return suffixArray.size();

    size_t left{0}; size_t right{reference.size()};
    size_t mlr = std::min(l,r);

    while (right - left > 1) {
        size_t middle{static_cast<size_t>(std::ceil((left+right)/2))};

        if (containsMLR(query, reference, suffixArray[middle], mlr) || biggerMLR(query, reference, suffixArray[middle], mlr)) {
            l = mlr;
            mlr = std::min(l,r);
            left = middle;
        } else {
            r = mlr;
            mlr = std::min(l,r);
            right = middle;
        }
    }

    return right;
}

bool binarySearchMLR (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                   const std::vector<saidx_t> & suffixArray, Borders & borders) {

    borders.left = leftBorderMLR(query, reference, suffixArray);
    borders.right = rightBorderMLR(query, reference, suffixArray);

    return borders.left < borders.right;
}

size_t computeLCP(saidx_t left, saidx_t right, const  std::vector<seqan3::dna5> & reference) {

    size_t lcp{0};

    while (static_cast<size_t>(left) < reference.size() && static_cast<size_t>(right) < reference.size() &&
            reference[left] == reference[right]) {
        ++left; ++right; ++lcp;
    }

    return lcp;
}

size_t doRecursion(std::map<std::pair<size_t, size_t>, size_t> & lcpValues, const size_t left, const size_t right) {

    assertm(left < right, "left >= right\n");

    if (right - left == 2) {
        return std::min(lcpValues[std::make_pair(left, left+1)], lcpValues[std::make_pair(right-1, right)]);
    }

    if (right - left == 1) {
        return lcpValues[std::make_pair(left, right)];
    }

    auto middle = static_cast<size_t>(std::ceil((left+right)/2));

    size_t leftLCP = doRecursion(lcpValues, middle, right);
    size_t rightLCP = doRecursion(lcpValues,left,  middle);

    lcpValues.emplace(std::make_pair(middle, right), leftLCP);
    lcpValues.emplace(std::make_pair(left, middle), rightLCP);

    return std::min(leftLCP, rightLCP);
}

void calculateInnerLCPValues(std::map<std::pair<size_t, size_t>, size_t> & lcpValues, const size_t size) {

    lcpValues.emplace(std::make_pair(0, size-1), doRecursion(lcpValues, 0, size-1));
}

void computeLCPValues (const std::vector<saidx_t> & suffixArray, const std::vector<seqan3::dna5> & reference,
                       std::map<std::pair<size_t, size_t>, size_t> & lcpValues) {

    for (size_t i = 1; i < suffixArray.size(); ++i) {
        lcpValues.emplace(std::make_pair(i-1, i), computeLCP(suffixArray[i-1], suffixArray[i], reference));
    }

    calculateInnerLCPValues(lcpValues, suffixArray.size());
}

bool compareSequence(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
                     const saidx_t position,  size_t &mlr ) {
    for (size_t i = mlr; i < std::min(left.size(), right.size()-position); ++i) {
        if (left[i] == right[position+i]) {
            ++mlr;
            continue;
        } else {
            return left[i] <= right[position+i];
        }
    }
    return left.size() <= right.size()-position;
}

bool writeLCPValues(const std::string & lcpOutFile, const std::map<std::pair<size_t,size_t>, size_t> & lcpValues) {
    // Initialize output stream
    std::ofstream fout{lcpOutFile, std::ios::binary | std::ios::out};

    // CHECK: if output stream is open
    if (!fout.is_open()) {
        std::cerr << "[WRITE SA] Can't write file: " << lcpOutFile << "\n";
        return false;
    }
    auto size = static_cast<std::streamsize>(lcpValues.size());
    fout.write(reinterpret_cast<char *>(&size), sizeof(size));

    for (std::pair<std::pair<size_t, size_t>, size_t> entry: lcpValues) {
        fout.write(reinterpret_cast<char *>(&entry.first.first), sizeof(entry.first.first));
        fout.write(reinterpret_cast<char *>(&entry.first.second), sizeof(entry.first.second));
        fout.write(reinterpret_cast<char *>(&entry.second), sizeof(entry.second));
    }

    // Close the output stream
    fout.close();
    return true;
}

bool readLCPValues(const std::string & lcpInFile, std::map<std::pair<size_t,size_t>, size_t> & lcpValues) {
    // Initialize input stream
    std::ifstream  fin{lcpInFile, std::ios::binary | std::ios::in};

    // CHECK: if its open
    if (!fin.is_open()) {
        std::cerr << "Can't read file: " << lcpInFile << "\n";
        return false;
    }

    // Read suffix array
    {
        std::streamsize sa_size{};
        fin.read(reinterpret_cast<char *>(&sa_size), sizeof(sa_size));

        for (size_t i = 0; i < static_cast<size_t>(sa_size); ++i) {

            std::pair<size_t, size_t> positions{};
            size_t lcpValue{};

            fin.read(reinterpret_cast<char *>(&positions.first), sizeof(positions.first));
            fin.read(reinterpret_cast<char *>(&positions.second), sizeof(positions.second));
            fin.read(reinterpret_cast<char *>(&lcpValue), sizeof(lcpValue));

            lcpValues[positions] = lcpValue;
        }
    }

    // Close the input stream
    fin.close();
    return true;
}

bool smallerEqualLCP(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
                     const saidx_t position, const size_t l, const  size_t r, const size_t L, const size_t M,
                     const size_t R ,size_t &mlr,std::map<std::pair<size_t, size_t>, size_t> & lcpValues) {

    if (l > r) {

        size_t lcpValue = lcpValues[std::make_pair(L,M)];

        // case 1 lcp(L,M) > l => Take L
        if (lcpValue > l) {
            return false;
        }

        // case 2 lcp(L,M) < l => Take R
        if (lcpValue < l) {
            return true;
        }

        // case 3 lcp(L,M) = l => compare sequences
        mlr = lcpValue;
        return compareSequence(left, right, position, mlr);

    } else if (l < r)  {

        size_t lcpValue = lcpValues[std::make_pair(M, R)];

        // case 1 lcp(L,M) > r => Take R
        if (lcpValue > r) {
            return true;
        }

        // case 2 lcp(L,M) < r => Take L
        if (lcpValue < r) {
            return false;
        }

        // case 3 lcp(L,M) = r => compare sequences
        mlr = lcpValue;
        return compareSequence(left, right, position, mlr);

    } else {
        return compareSequence(left, right, position, mlr);
    }
}

size_t leftBorderLCP (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                      const std::vector<saidx_t> & suffixArray, std::map<std::pair<size_t, size_t>, size_t> & lcpValues){

    size_t l{0};  size_t r{0};

    // Since l = r => no advantage of using lcp. Using mlr instead
    if (smallerMLR(query, reference, suffixArray[0], l)) return 0;
    if (!smallerEqualMLR(query, reference, suffixArray[suffixArray.size()-1], r)) return suffixArray.size();

    size_t left{0}; size_t right{reference.size()};
    size_t mlr = std::min(l,r);

    while (right - left > 1) {
        size_t middle{static_cast<size_t>(std::ceil((left+right)/2))};

        if (smallerEqualLCP(query, reference, suffixArray[middle], l, r, left, middle, right, mlr, lcpValues)) {
            r = mlr;
            mlr = std::min(l,r);
            right = middle;
        } else {
            l = mlr;
            mlr = std::min(l,r);
            left = middle;
        }
    }

    if (smallerEqual(query,reference, suffixArray[left])) {
        return left;
    } else {
        return right;
    }
}

bool biggerLCP(const std::vector<seqan3::dna5> & left, const std::vector<seqan3::dna5> & right,
               const saidx_t position, const size_t l, const  size_t r, const size_t L, const size_t M,
               const size_t R ,size_t &mlr,std::map<std::pair<size_t, size_t>, size_t> & lcpValues) {

    return (!smallerEqualLCP(left, right, position, l, r, L, M, R, mlr, lcpValues));
}

size_t rightBorderLCP (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                       const std::vector<saidx_t> & suffixArray, std::map<std::pair<size_t, size_t>, size_t> & lcpValues){

    size_t l{0};  size_t r{0};

    if (smallerMLR(query, reference, suffixArray[0], l)) return 0;
    if (contains(query, reference, suffixArray[suffixArray.size()-1])) return suffixArray.size();

    size_t left{0}; size_t right{reference.size()};
    size_t mlr = std::min(l,r);

    while (right - left > 1) {
        size_t middle{static_cast<size_t>(std::ceil((left+right)/2))};

        if (containsMLR(query, reference, suffixArray[middle], mlr) ||
                biggerLCP(query, reference, suffixArray[middle], l, r, left, middle, right, mlr, lcpValues)) {
            l = mlr;
            mlr = std::min(l,r);
            left = middle;
        } else {
            r = mlr;
            mlr = std::min(l,r);
            right = middle;
        }
    }

    return right;
}


bool binarySearchLCP (const std::vector<seqan3::dna5> & query, const std::vector<seqan3::dna5> & reference,
                      const std::vector<saidx_t> & suffixArray, Borders & borders,
                      std::map<std::pair<size_t, size_t>, size_t> & lcpValues) {

    borders.left = leftBorderLCP(query, reference, suffixArray, lcpValues);
    //// To-do Implement right border LCP smarter
    borders.right = rightBorderLCP(query, reference, suffixArray, lcpValues);

    return borders.left < borders.right;
}

void findOccurrences(const Borders & borders, const std::vector<saidx_t> & suffixarray, std::vector<saidx_t> & occurrences) {


    //// TO-DO: Why does reserve produces an error? Is there an other way?
    // Produces an error
    // occurrences.reserve(borders.left - borders.right);

    for (size_t i = borders.left; i < borders.right; ++i) {
        occurrences.push_back(suffixarray[i]);
    }
}
void print( const std::vector<saidx_t> & suffixArray) {
    for (const saidx_t suff_tab : suffixArray) {
        std::cout << suff_tab << "\n";
    }
}

void runTestSearchWithMLR() {
    seqan3::debug_stream << "RUN TEST\n";

    std::vector<seqan3::dna5> testReference{"AAAATACCCAAATAACACACACAAAATAATCATCATCATCATCATCATCCAGGGTACACACGTATGACAACGTTACCGACTACACACACA"_dna5};

    std::vector<saidx_t> suffixarray;
    suffixarray.resize(testReference.size()); // resizing the array, so it can hold the complete SA

    auto sa_str = reinterpret_cast<sauchar_t const*>(testReference.data());

    divsufsort(sa_str, suffixarray.data(), static_cast<saidx_t>(testReference.size()));

    Borders borders{};

    std::cout << "Search 'A' in ACACACACA\n";

    binarySearchMLR("A"_dna5, testReference,suffixarray, borders);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";
}

void runTestSearchWithLCP() {
    seqan3::debug_stream << "RUN TEST\n";

    std::vector<seqan3::dna5> testReference{"GATAGACA"_dna5};

    std::vector<saidx_t> suffixArray;
    suffixArray.resize(testReference.size()); // resizing the array, so it can hold the complete SA

    auto sa_str = reinterpret_cast<sauchar_t const*>(testReference.data());

    divsufsort(sa_str, suffixArray.data(), static_cast<saidx_t>(testReference.size()));

    std::map<std::pair<size_t, size_t>, size_t> lcpValues{};
    computeLCPValues(suffixArray, testReference, lcpValues);

    for (auto lcpValue : lcpValues) {
        std::cout << lcpValue.first.first << "\t" << lcpValue.first.second << "\t" << lcpValue.second << "\n";
    }

    Borders borders{};

    std::cout << "Search 'ACA' in GATAGACA\n";

    binarySearchLCP("A"_dna5, testReference,suffixArray, borders, lcpValues);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";
}

void runTestSearchSA() {
    seqan3::debug_stream << "RUN TEST\n";

    std::vector<seqan3::dna5> testReference{"GATAGACA"_dna5};

    std::vector<saidx_t> suffixarray;
    suffixarray.resize(testReference.size()); // resizing the array, so it can hold the complete SA

    auto sa_str = reinterpret_cast<sauchar_t const*>(testReference.data());

    divsufsort(sa_str, suffixarray.data(), static_cast<saidx_t>(testReference.size()));

    print(suffixarray);

    Borders borders{};

    std::cout << "Search 'A' in GATAGACA\n";

    binarySearch("A"_dna5, testReference,suffixarray, borders);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";

    std::cout << "Search 'TAGACA' in GATAGACA\n";

    binarySearch("TAGACA"_dna5, testReference,suffixarray, borders);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";

    std::cout << "Search 'ACA' in GATAGACA\n";

    binarySearch("ACA"_dna5, testReference,suffixarray, borders);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";

    std::cout << "Search 'C' in GATAGACA\n";

    binarySearch("C"_dna5, testReference,suffixarray, borders);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";

    std::cout << "Search 'GATAGACA' in GATAGACA\n";

    binarySearch("GATAGACA"_dna5, testReference,suffixarray, borders);

    std::cout << "Left border: "<< borders.left << "\n";
    std::cout << "Right border: " << borders.right << "\n";
}

int main(int argc, char const* const* argv) {

    bool runLCPSearch{false};

    bool testSearchSA{false};
    bool testSearchWithML{false};
    bool testSearchWithLCP{false};

    seqan3::argument_parser parser{"suffixarray_search", argc, argv, seqan3::update_notifications::off};

    parser.info.author = "SeqAn-Team";
    parser.info.version = "1.0.0";

    auto reference_file = std::filesystem::path{};
    parser.add_option(reference_file, '\0', "reference", "path to the reference file");

    auto query_file = std::filesystem::path{};
    parser.add_option(query_file, '\0', "query", "path to the query file");

    std::string output_file;
    parser.add_option(output_file, '\0', "out", "path to output file");

    bool calculatingLCPValues{false};
    parser.add_flag(calculatingLCPValues, 'l', "calculating-lcp-values", "Set to calculate lcp values");

    size_t nrOfRuns{1};
    parser.add_option(nrOfRuns, 'i', "iterations", "Number of iterations");

    size_t nrOfQueries{0};
    parser.add_option(nrOfQueries, 'n', "number-of-queries", "Number of queries");

    std::string lcpOutFile{};
    parser.add_option(lcpOutFile, '\0', "lcp-out", "path to lcp output file");

    std::string lcpInFile{};
    parser.add_option(lcpOutFile, '\0', "lcp-in", "path to lcp input file");

    try {
        parser.parse();
    } catch (seqan3::argument_parser_error const &ext) {
        seqan3::debug_stream << "Parsing error. " << ext.what() << "\n";
        return EXIT_FAILURE;
    }

    if (testSearchSA) {
        runTestSearchSA();
    }
    if (testSearchWithML) {
        runTestSearchWithMLR();
    }
    if (testSearchWithLCP) {
        runTestSearchWithLCP();
    }

    std::cout << "Load data\n";

    // loading our files
    auto reference_stream = seqan3::sequence_file_input{reference_file};
    auto query_stream = seqan3::sequence_file_input{query_file};

    // read reference into memory
    // Attention: we are concatenating all sequences into one big combined sequence
    //            this is done to simplify the implementation of suffix_arrays
    std::vector<seqan3::dna5> reference;
    for (auto &record: reference_stream) {
        auto r = record.sequence();
        reference.insert(reference.end(), r.begin(), r.end());
    }

    // read query into memory
    std::vector<std::vector<seqan3::dna5>> queries;
    for (auto &record: query_stream) {
        queries.push_back(record.sequence());
    }
    if (nrOfQueries < queries.size()) {
        if (nrOfQueries != 0) {
            std::cout << "Set number queries from " << queries.size() << " possible queries to " << nrOfQueries << "\n";
            queries.resize(nrOfQueries); // will reduce the amount of searches
        }
    } else {
            std::cout << "No more queries then " << queries.size() << "!\n";
    }

    // Array that should hold the future suffix array
    std::vector<saidx_t> suffixArray;
    suffixArray.resize(reference.size()); // resizing the array, so it can hold the complete SA

    auto sa_str = reinterpret_cast<sauchar_t const *>(reference.data());
    std::cout << "Constructing suffix array\n";
    divsufsort(sa_str, suffixArray.data(), static_cast<saidx_t>(reference.size()));

    std::map<std::pair<size_t, size_t>, size_t> lcpValues{};

    if (calculatingLCPValues && !lcpInFile.empty()) {
        std::cout << "-l flag is set to calculate, as well as --lcp-in to load lcp values\n";
        std::cout << "=> loaded one will be used!";
    } else if (!calculatingLCPValues && !lcpOutFile.empty()) {
        std::cout << "-l flag is not set to calculate, but --lcp-out to save lcp values\n";
        std::cout << "=> lcp values will be calculated but not used!";
    }

    if (calculatingLCPValues) runLCPSearch = true;

    if (calculatingLCPValues || !lcpOutFile.empty()) {
        std::cout << "Calculating LCP values\n";

        computeLCPValues(suffixArray, reference, lcpValues);

        if (!lcpOutFile.empty()) {
            std::cout << "Write LCP values to " << lcpOutFile << "\n";
            writeLCPValues(lcpOutFile,lcpValues);
        }
    }

    if (!lcpInFile.empty()) {
        runLCPSearch = true;

        std::cout << "Load LCP-values\n";
        readLCPValues(lcpInFile, lcpValues);
    }

    std::ofstream ofstream(output_file);
    ofstream << "BS,MLR";

    if (runLCPSearch) {
        ofstream <<"LCP\n";
    } else {
        ofstream << "\n";
    }

    for (size_t i = 0; i < nrOfRuns; ++i) {
        std::cout << "Start Iteration " << i << "\n";
        auto start = std::chrono::system_clock::now();

        for (const auto &q: queries) {

            Borders borders{};

            std::vector<saidx_t> occurrences{};

            binarySearch(q, reference, suffixArray, borders);

            findOccurrences(borders, suffixArray, occurrences);
        }

        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Simple binary search required " << elapsed.count() << " seconds to run\n";

        ofstream << elapsed.count() << ",";

        start = std::chrono::system_clock::now();

        for (const auto &q: queries) {

            Borders borders{};

            std::vector<saidx_t> occurrences{};

            binarySearchMLR(q, reference, suffixArray, borders);

            findOccurrences(borders, suffixArray, occurrences);
        }

        end = std::chrono::system_clock::now();

        elapsed = end - start;

        std::cout << "Binary search with MLR required " << elapsed.count() << " seconds to run\n";

        if (runLCPSearch) {

            ofstream << elapsed.count() << ",";

            start = std::chrono::system_clock::now();

            for (const auto &q: queries) {

                Borders borders{};

                std::vector<saidx_t> occurrences{};

                binarySearchLCP(q, reference, suffixArray, borders, lcpValues);

                findOccurrences(borders, suffixArray, occurrences);
            }

            end = std::chrono::system_clock::now();

            elapsed = end - start;

            std::cout << "Binary search with LCP required " << elapsed.count() << " seconds to run\n";
        }
        ofstream << elapsed.count() << "\n";
    }

    ofstream.close();

    return 0;
}
