#include <iostream>
using namespace std;
class Solution {
public:
    bool isPowerOfTwo(int n) {
        if(n==1) return true;
        while(n>=2){
            if(n==2) return true;
            if(n<2)  return false;
            if(n%2 !=0) return false;  // if n is odd, it cannot be a power of two
            n=(n/2) ;
        }
        return false;
    }
};