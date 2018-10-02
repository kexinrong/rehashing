//
// Created by Kexin Rong on 10/2/18.
//

#ifndef HBE_PARSECONFIG_H
#define HBE_PARSECONFIG_H

#include <config4cpp/Configuration.h>
using namespace config4cpp;

class parseConfig {
public:
    Configuration *cfg = Configuration::create();
    const char* scope;

    parseConfig(const char * configFile, const char* s) {
        cfg->parse(configFile);
        scope = s;
    }

    int getDim() {
        return cfg->lookupInt(scope, "d");
    }

    int getN() {
        return cfg->lookupInt(scope, "n");
    }

    double getH() {
        return cfg->lookupFloat(scope, "h");
    }

    double getEps() {
        return cfg->lookupFloat(scope, "eps");
    }

    double getTau() {
        return cfg->lookupFloat(scope, "tau");
    }

    double getBeta() {
        return cfg->lookupFloat(scope, "beta");
    }

    double getSampleRatio() {
        return cfg->lookupFloat(scope, "sample_ratio");
    }

    int getSamples() {
        return cfg->lookupInt(scope, "samples");
    }

    int getStartCol() {
        return cfg->lookupInt(scope, "start_col");
    }

    int getEndCol() {
        return cfg->lookupInt(scope, "end_col");
    }

    bool ignoreHeader() {
        return cfg->lookupBoolean(scope, "ignore_header");
    }

    const char* getDataFile() {
        return cfg->lookupString(scope, "fpath");
    }

    const char* getExactPath() {
        return cfg->lookupString(scope, "exact_path");
    }

    const char* getKernel() {
        return cfg->lookupString(scope, "kernel");
    }

};


#endif //HBE_PARSECONFIG_H
