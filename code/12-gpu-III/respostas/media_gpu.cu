#include "../imagem.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <iostream>
#include <string>

struct filtro_media {
    unsigned char *img;
    int rows, cols, total;
    
    filtro_media(int r, int c, unsigned char *img): img(img), rows(r), cols(c) {
        total = r * c;
    };

    __device__
    unsigned char operator()(const int &k) {
        int s = img[k];
        if (k - cols >= 0) 
            s += img[k - cols];
        if (k + cols < total)
            s += img[k + cols];
        if (k + 1 < total) 
            s += img[k + 1];
        if (k - 1 >= 0)
            s += img[k - 1];

        return s / 5;
    }
};

int main(int argc, char *argv[]) {
    std::string entrada = argv[1];

    imagem *i = read_pgm(entrada);

    thrust::device_vector<unsigned char> img(
        i->pixels,
        i->pixels + i->total_size
    );

    thrust::device_vector<unsigned char> out(i->total_size);

    filtro_media fm(i->rows, i->cols, thrust::raw_pointer_cast(img.data()));
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(i->total_size),
                      out.begin(),
                      fm);

    thrust::copy(out.begin(), out.end(), i->pixels);
    write_pgm(i, argv[2]);
    return 0;
}