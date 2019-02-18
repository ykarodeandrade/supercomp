// Exemplo de uso de recursos do Boost
// Compilar: g++ serializar.cpp -std=c++14 -lboost_serialization
// Referência: https://www.quantlib.org/slides/dima-boost-intro.pdf
// Autor: Luciano Soares <lpsoares@insper.edu.br>

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

int main(int argc, const char *argv[]) {

    // Faz a analise dos dados passados na linha de comando
    boost::program_options::options_description desc{"Options"};
    desc.add_options() ("gravar,g", "grava dados") ("ler,l", "le dados");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    // Inicia um timer e define o arquivo para salvar os dados
    boost::timer t;
    std::string filename("./normal.bin");

    if (vm.count("gravar")) { // cria números aleatórios e salva em um arquivo

        // cria o stream de saida com o apoio do Boost
        std::ofstream ostr(filename.c_str(),std::ios::binary);
        boost::archive::binary_oarchive oa(ostr);

        // cria sistema de números aleatórios em uma distribuição normal
        unsigned long seed = 89210;
        boost::mt19937 rng(seed);
        boost::normal_distribution <> norm;
        boost::variate_generator<boost::mt19937&, boost::normal_distribution <>> normGen(rng,norm);

        // gera os números aleatórios em um vetor
        int numVars = 1000000;
        std::vector<double> myVec(numVars);
        BOOST_FOREACH(double& x, myVec) x=normGen();

        // envia os dados para o arquivo
        oa << myVec;
        std::cout << "Dados gravados" << std::endl;

        // fecha o arquivo que recebeu os dados
        ostr.close();
        
    } else if (vm.count("ler")) { // faz a leitura de números aleatórios de um arquivo

        // cria o stream de entrada de dados com o apoio do Boost
        std::ifstream istr(filename.c_str(), std::ios::binary);
        boost::archive::binary_iarchive ia(istr);

        // cria um vetore e preenche ele com os dados do arquivo
        std::vector <double > myVecLoaded;
        ia >> myVecLoaded;

        // fecha o arquivo que forneceu os dados
        istr.close();

        // exibe somente os 10 primeiros valores lidos
        for(int i=0; i<10;i++) std::cout << myVecLoaded[i] << std::endl;

        std::cout << "Dados lidos" << std::endl;    
    }

    // Exibe o tempo decorrido
    std::cout << "Tempo decorrido:" << t.elapsed() << std::endl;

}