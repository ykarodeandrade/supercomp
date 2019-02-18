#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream> 
#include <boost/timer.hpp>
#include <boost/foreach.hpp>
#include <boost/random.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>  
#include <boost/program_options.hpp>

using namespace boost::program_options;

int main(int argc, const char *argv[]) {

    options_description desc;
    desc.add_options()
      ("pos", value<int>()->default_value(1));
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);

    int posicao = vm["pos"].as<int>();
    bool regrava = false;
    std::vector<long int> myVec;
    std::string filename("./fibonacci.bin");

    std::ifstream istr(filename.c_str(), std::ios::binary);
    if (istr) {
        boost::archive::binary_iarchive ia(istr);
        ia >> myVec;
        std::cout << myVec.size() << "registros recuperados\n";
    } else {
        regrava=true;
        myVec = {0,1};
    }
    istr.close();

    int tmpPos = myVec.size();
    if(posicao>tmpPos) regrava=true;
    while(posicao > (tmpPos = myVec.size())) {
        myVec.push_back(myVec[tmpPos-1]+myVec[tmpPos-2]);
    }

    if(regrava) {
        std::ofstream ostr(filename.c_str(),std::ios::binary);
        boost::archive::binary_oarchive oa(ostr);
        oa << myVec;
        ostr.close();
    }

    std::cout << "Fibonacci[" << posicao << "] = " << myVec[posicao-1] << std::endl;

}