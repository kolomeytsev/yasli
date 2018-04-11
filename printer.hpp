#ifndef printer_hpp
#define printer_hpp

#include <iostream>
#include <fstream>
#include <vector>

std::vector< std::pair<std::vector<float>, float> > ReadData();
void PrintVector(const std::vector<float>& vec);
void PrintMatrixconst (const std::vector<std::vector<float>>& mat);
void PrintDataconst (const std::vector< std::pair<std::vector<float>, float> >& data);

#include "printer.cpp"
#endif // printer_hpp
