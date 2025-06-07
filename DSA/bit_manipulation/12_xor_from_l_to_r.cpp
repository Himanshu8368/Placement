#include <iostream>
using namespace std;
class Solution{	
	public:
		int findRangeXOR(int l,int r){
			int Xor;
            for(int i=l;i<=r;i++){
                Xor=Xor ^i;
            }
            return Xor;
		}
};