//
//  Created by YDuan on 2/10/2019.
//  Copyright © 2019 Duan. All rights reserved.
//

#ifndef INPUT_H
#define INPUT_H

#include <stdio.h>
#include <string>

class Input {


    public:
        int narg;
        char **arg;

        Input(int, char **);
        Input()~;
        void file();                //process all input
        void file(const char *);    //process an input script
        char *one(const char *);    //process a single command
        void substitute()

    protected:
        char *command;
        int maxarg;
        char *line, *copy, *work;
	

}
