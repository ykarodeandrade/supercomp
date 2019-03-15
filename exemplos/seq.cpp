#include <iostream>
#include "seq.h"

seq::seq() {x=2;}

void seq::pot() {x*=2;}

int seq::num() {
    pot();
    return(x);
}
