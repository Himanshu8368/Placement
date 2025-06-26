#include <vector>
using namespace std;
class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        vector<int> arr(nums.size(),0);
        int pos_index =0;
        int negative_index = 1;
        for(int i=0;i<nums.size();i++){
            if(nums[i]>0){
                arr[pos_index] = nums[i];
                pos_index+=2;
            }
            else{
                arr[negative_index] = nums[i];
                negative_index+=2;
            }
        }
        return arr;
    }
};