/*! \file
    \brief Public interfaces for function(s) which takes a list of user-specified 
    function objects and adds them to the ModelObject.

 */

#ifndef _ADD_FUNCTION_H_
#define _ADD_FUNCTION_H_

#include <string>
#include <vector>
#include "model_object.h"

using namespace std;

// static vector< map<string, string> > EMPTY_MAP_VECTOR;
static vector<bool> EMPTY_BOOL_VECTOR;


// NOTE: (some of) the following functions are used in PyImfit

//! Main function which adds a list of FunctionObject instances to an instance of ModelObject
int AddFunctions( ModelObject *theModel, const vector<string> &functionNameList,
                  vector<string> &functionLabelList, vector<int> &functionSetIndices, 
                  const bool subamplingFlag, const int verboseFlag=0, 
                  vector< map<string, string> > &extraParams=EMPTY_MAP_VECTOR,
                  const vector<bool> &globalFuncFlags=EMPTY_BOOL_VECTOR );

//! Prints out names of available image functions (FunctionObject classes) to stdout
void PrintAvailableFunctions( );

//! Prints out list of available functions and their parameters
/*! Use this to print out a full list consisting of each function
 * name ("FUNCTION <short-name>") followed by the ordered list of parameter
 * names (suitable for copying and pasting into a config file for makeimage or imfit).
 */
void ListFunctionParameters( );

//! Populates a vector with names of the specified function's parameters
int GetFunctionParameterNames( string &functionName, vector<string> &parameterNameList );

//! Populates a vector with names of available functions (FunctionObject classes)
void GetFunctionNames( vector<string> &functionNameList );


#endif  // _ADD_FUNCTION_H_
