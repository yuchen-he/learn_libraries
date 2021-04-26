#include "test1/test1.h"
#include <iostream>

using namespace std;

Test1::Test1(int num) {
	m_num = num;
}

int Test1::get_num() {
	return m_num;
}
