#include <iostream>
#include "seq.h"

int main() {
    seq *s = new seq();
    for(int i=0;i<10;i++) {
        std::cout << s->num() << std::endl;
    }
    delete s;
}
