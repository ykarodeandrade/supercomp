#include <iostream>
#include "boost/filesystem.hpp" 

using namespace boost::filesystem;

int main(int argc, char **argv) {
    
    path p(".");
    
    for (auto &entry : directory_iterator(p)) {
        std::cout << entry.path().string() << is_regular_file(entry.path()) << "\n";
        
    }
    
    return 0;
}
