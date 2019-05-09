// Copyright (C) 2003 Ronan Collobert (collober@idiap.ch)
//                
// This file is part of Torch 3.
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "CmdLine.h"
#include <ctime>

using namespace std;

namespace Torch {

// Oy. J'ai fait le menage dans c'te classe.
// Pleins de features non documentees dans le tutorial!

CmdLine::CmdLine()
{
  n_master_switches = 1; // the default!
  n_cmd_options.push_back(0);
  cmd_options.resize(1);
  text_info = NULL;
  working_directory = new char[2];
  strcpy(working_directory, ".");
  associated_files.resize(1);
  n_associated_files = 0;
  master_switch = -1;
  program_name = new char[1];
  *program_name = '\0';

}

void CmdLine::addInfo(const char *text)
{
	if(NULL != text_info){
		delete text_info;
		text_info = NULL;
	}
	text_info = new char[strlen(text) + 1];

	strcpy(text_info, text);
}

void CmdLine::addCmdOption(CmdOption *option)
{
	if(option->isMasterSwitch())
	{
		n_cmd_options.push_back(0);
		cmd_options.resize(n_master_switches+1);
		n_master_switches++;
	}

	int n = n_master_switches-1;
	cmd_options[n].push_back(option);
	n_cmd_options[n]++;
}

void CmdLine::addMasterSwitch(const char *text)
{
	CmdOption *option = new CmdOption(text, "", "", false);
	option->isMasterSwitch(true);
	addCmdOption(option);
}

void CmdLine::addICmdOption(const char *name, int *ptr, int init_value, const char *help, bool save_it)
{
	IntCmdOption *option = new IntCmdOption(name, ptr, init_value, help, save_it);
	addCmdOption(option);
}

void CmdLine::addLCmdOption(const char *name, long *ptr, long init_value, const char *help, bool save_it)
{
	LongCmdOption *option = new LongCmdOption(name, ptr, init_value, help, save_it);
	addCmdOption(option);
}

void CmdLine::addBCmdOption(const char *name, bool *ptr, bool init_value, const char *help, bool save_it)
{
	BoolCmdOption *option = new BoolCmdOption(name, ptr, init_value, help, save_it);
	addCmdOption(option);
}

void CmdLine::addRCmdOption(const char *name, real *ptr, real init_value, const char *help, bool save_it)
{
	RealCmdOption *option = new RealCmdOption(name, ptr, init_value, help, save_it);
	addCmdOption(option);
}

void CmdLine::addSCmdOption(const char *name, char **ptr, const char *init_value, const char *help, bool save_it)
{
	StringCmdOption *option = new StringCmdOption(name, ptr, init_value, help, save_it);
	addCmdOption(option);
}

void CmdLine::addICmdArg(const char *name, int *ptr, const char *help, bool save_it)
{
	IntCmdOption *option = new IntCmdOption(name, ptr, 0, help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addBCmdArg(const char *name, bool *ptr, const char *help, bool save_it)
{
	BoolCmdOption *option = new BoolCmdOption(name, ptr, false, help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addRCmdArg(const char *name, real *ptr, const char *help, bool save_it)
{
	RealCmdOption *option = new RealCmdOption(name, ptr, 0., help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addSCmdArg(const char *name, char **ptr, const char *help, bool save_it)
{
	StringCmdOption *option = new StringCmdOption(name, ptr, "", help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addText(const char *text)
{
	CmdOption *option = new CmdOption(text, "", "", false);
	option->isText(true);
	addCmdOption(option);
}

int CmdLine::read(int argc_, char **argv_)
{
	if(NULL != program_name){
		delete[] program_name;
		program_name = NULL;
	}

	program_name = new char[strlen(argv_[0])+1];
	strcpy(program_name, argv_[0]);
	argv = argv_+1;
	argc = argc_-1;
  
  // Look for help request and the Master Switch
	master_switch = 0;
	if(argc >= 1)
	{
		if( ! (strcmp(argv[0], "-h") && strcmp(argv[0], "-help") && strcmp(argv[0], "--help")) )
		help();

		for(int i = 1; i < n_master_switches; i++)
		{
			if(cmd_options[i][0]->isCurrent(&argc, &argv))
			{
				master_switch = i;
				break;
			}
		}
	}
  
	vector<CmdOption*> cmd_options_ = cmd_options[master_switch];
	int n_cmd_options_ = n_cmd_options[master_switch];

	// Initialize the options.
	for(int i = 0; i < n_cmd_options_; i++)
	{
		cmd_options_[i]->initValue();
	}

	while(argc > 0)
	{
		// First, check the option.
		int current_option = -1;    
		for(int i = 0; i < n_cmd_options_; i++)
		{
			if(cmd_options_[i]->isCurrent(&argc, &argv))
			{
				current_option = i;
				break;
			}
		}

		if(current_option >= 0)
		{
			if(cmd_options_[current_option]->is_setted)
				error("CmdLine: option %s is setted twice", cmd_options_[current_option]->name);
			cmd_options_[current_option]->read(&argc, &argv);
			cmd_options_[current_option]->is_setted = true;
		}
		else
		{
			// Check for arguments
			for(int i = 0; i < n_cmd_options_; i++)
			{
				if(cmd_options_[i]->isArgument() && (!cmd_options_[i]->is_setted))
				{
					current_option = i;
					break;
				}
			}
       
			if(current_option >= 0)
			{
				cmd_options_[current_option]->read(&argc, &argv);
				cmd_options_[current_option]->is_setted = true;        
			}
			else
				error("CmdLine: parse error near <%s>. Too many arguments.", argv[0]);
		}    
	}

	// Check for empty arguments
	for(int i = 0; i < n_cmd_options_; i++)
	{
		if(cmd_options_[i]->isArgument() && (!cmd_options_[i]->is_setted))
		{
			message("CmdLine: not enough arguments!\n");
			help();
		}
	}

	if(write_log)
	{
		//DiskXFile cmd_log("cmd.log", "w");
		//writeLog(&cmd_log, false);
	}
	cmd_options_.clear();

	return master_switch;
}

// RhhAHha AH AHa hha hahaAH Ha ha ha

void CmdLine::help()
{
  if(text_info)
    print("%s\n", text_info);

  for(int master_switch_ = 0; master_switch_ < n_master_switches; master_switch_++)
  {
    int n_cmd_options_ = n_cmd_options[master_switch_];
	vector<CmdOption*> cmd_options_ = cmd_options[master_switch_];

    int n_real_options = 0;
    for(int i = 0; i < n_cmd_options_; i++)
    {
      if(cmd_options_[i]->isOption())
        n_real_options++;
    }

    if(master_switch_ == 0)
    {
      print("#\n");
      print("# usage: %s", program_name);
      if(n_real_options > 0)
        print(" [options]");
    }
    else
    {
      print("\n#\n");
      print("# or: %s %s", program_name, cmd_options_[0]->name);
      if(n_real_options > 0)
        print(" [options]");
    }

    for(int i = 0; i < n_cmd_options_; i++)
    {
      if(cmd_options_[i]->isArgument())
        print(" <%s>", cmd_options_[i]->name);
    }
    print("\n#\n");

    // Cherche la longueur max du param
    int long_max = 0;
    for(int i = 0; i < n_cmd_options_; i++)
    {
      int laurence = 0;
      if(cmd_options_[i]->isArgument())
        laurence = (int)strlen(cmd_options_[i]->name)+2;

      if(cmd_options_[i]->isOption())
        laurence = (int)(strlen(cmd_options_[i]->name)+strlen(cmd_options_[i]->type_name)+1);
      
      if(long_max < laurence)
        long_max = laurence;
    }

    for(int i = 0; i < n_cmd_options_; i++)
    {
      int z = 0;
      if(cmd_options_[i]->isText())
      {
        z = -1;
        print("%s", cmd_options_[i]->name);
      }

      if(cmd_options_[i]->isArgument())
      {
        z = (int)strlen(cmd_options_[i]->name)+2;
        print("  ");
        print("<%s>", cmd_options_[i]->name);
      }
      
      if(cmd_options_[i]->isOption())
      {
        z = (int)(strlen(cmd_options_[i]->name)+strlen(cmd_options_[i]->type_name)+1);
        print("  ");
        print("%s", cmd_options_[i]->name);
        print(" %s", cmd_options_[i]->type_name);
      }

      if(z >= 0)
      {
        for(int i = 0; i < long_max+1-z; i++)
          print(" ");
      }
      
      if( cmd_options_[i]->isOption() || cmd_options_[i]->isArgument() )
        print("-> %s", cmd_options_[i]->help);
    
      if(cmd_options_[i]->isArgument())
        print(" (%s)", cmd_options_[i]->type_name);

      if(cmd_options_[i]->isOption())
      {
        //DiskXFile std_out(stdout);
        print(" ");
        //cmd_options_[i]->printValue(&std_out);
      }

      if(!cmd_options_[i]->isMasterSwitch())
        print("\n");
    }
  }  
  exit(-1);
}

void CmdLine::setWorkingDirectory(const char* dirname)
{
  //allocator->free(working_directory);
	if(NULL != working_directory){
		delete[] working_directory;
		working_directory = NULL;
	}
	//working_directory = (char *)allocator->alloc(strlen(dirname)+1);
	working_directory = new char[strlen(dirname)+1];
	strcpy(working_directory, dirname);
}

char *CmdLine::getPath(const char *filename)
{
	char *path_ = new char[strlen(working_directory)+strlen(filename)+2];
	strcpy(path_, working_directory);
	strcat(path_, "/");
	strcat(path_, filename);
	char *tmpName = new char[strlen(filename)+1];
	strcpy(tmpName, filename);
	associated_files.push_back(tmpName);
	n_associated_files++;
	return path_;
}

CmdLine::~CmdLine()
{
	if(NULL != program_name){
		delete[] program_name;
		program_name = NULL;
	}
	if(NULL != text_info){
		delete[] text_info;
		text_info = NULL;
	}
	if(NULL != working_directory){
		delete[] working_directory;
		working_directory = NULL;
	}
	vector<char*>::iterator itrChar;
	for(itrChar = associated_files.begin(); itrChar != associated_files.end(); itrChar++)
	{
		delete(*itrChar);
		*itrChar = NULL;
	}
	associated_files.clear();

	n_cmd_options.clear();
	vector<vector<CmdOption*> >::iterator itrOut;
	vector<CmdOption*>::iterator itrIn;
	for(itrOut = cmd_options.begin(); itrOut != cmd_options.end(); itrOut++)
	{
		for(itrIn = itrOut->begin(); itrIn != itrOut->end(); itrIn++)
		{
			delete(*itrIn);
			*itrIn = NULL;
		}
		itrOut->clear();
	}
	cmd_options.clear();
	
}

}
