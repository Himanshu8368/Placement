#include <vector>
#include <algorithm> 
using namespace std;

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int zero_index = -1;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==0){
                zero_index =i;
                break;
            }
        }

        if( zero_index ==-1){
            return;
        }
        for(int i = zero_index +1;i<nums.size();i++){
            if(nums[i]!=0){
                swap(nums[zero_index],nums[i]);
                zero_index++;
            }
        }


    }
};

// we use another approach to solve this problem in which we first move all the non-zero elements to the front of the array and then fill the rest with zeros
class Solution2 {
public:
    void moveZeroes(vector<int>& nums) {
        int non_zero_index = 0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]!=0){
                nums[non_zero_index++] = nums[i];
            }
        }
        for(int i=non_zero_index;i<nums.size();i++){
            nums[i] = 0;
        }
    }
};