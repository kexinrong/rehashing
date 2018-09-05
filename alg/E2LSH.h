//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_E2LSH_H
#define HBE_E2LSH_H


#include "BaseLSH.h"

class E2LSH : public BaseLSH {
public:
    std::vector<HashTable> tables;
};


#endif //HBE_E2LSH_H
