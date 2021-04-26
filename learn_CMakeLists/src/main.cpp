#include "test1/test1.h"
#include <iostream>

using namespace std;

int main() {
        //int num = 25;
	Test1 t(25);
	int num = t.get_num();
	std::cout << "num is: " << num << endl;
	return 0;
}
