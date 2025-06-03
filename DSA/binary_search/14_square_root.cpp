#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    long long floorSqrt(long long n)  {
      int low =0;
      int high = n;
      int ans = n;
      while(low<=high){
        int mid = (low + high)/2;
        int value = mid * mid;
        if(n >= value){
            ans = mid ;
            low = mid+1;
        }
        else{
            high = mid -1;
        }
      }
      return ans ;
    }
};