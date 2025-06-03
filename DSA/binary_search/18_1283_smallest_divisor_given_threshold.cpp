#include <vector>
#include <limits.h> 
using namespace std;
class Solution {
public:
    int smallestDivisor(vector<int>& nums, int threshold) {

        int max = INT_MIN;
        for(int i=0;i<nums.size();i++){
            if(nums[i]>max){
                max = nums[i];
            }
        }
        int ans = threshold ;
        int low =1 ; int high = max;
        
        while(low<=high){
            int mid = (low+high)/2;
            int count =0;
            for(int i=0;i<nums.size();i++){
                count = count + nums[i]/mid;
                if(nums[i]%mid !=0){
                    count++;
                }
                // here we can also use count += (nums[i] + mid - 1) / mid; to avoid the if condition
            }
            if(count<= threshold){
                high = mid -1;
                ans = mid ;
            }
            else{
                low =  mid +1;

            }
        }
        return ans;
        
    }
};