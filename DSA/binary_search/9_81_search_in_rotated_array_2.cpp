#include <vector>
using namespace std;
// if(nums[low]==nums[mid] && nums[mid]==nums[high]){
//     low++;
//     high--;
// }
// this condition is used to handle duplicates in the array
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        bool ans = false;
        int low =0; int high = nums.size()-1;
        while(low<=high){
            int mid = (low+high)/2;
            if(nums[mid]== target) {
                ans = true;
                return ans ;
            }
            if(nums[low]==nums[mid] && nums[mid]==nums[high]){
                low++;
                high--;
            }
            else if(nums[high]>=nums[mid]){
                if(nums[high]>= target && nums[mid]<target){
                    low = mid+1;
                }
                else{
                    high = mid -1;
                }
            }
            else if(nums[mid]>=nums[low]){
                if(nums[mid]>target && nums[low]<=target){
                    high = mid -1;
                }
                else{
                    low = mid +1;
                }
            }
        }
        return ans ;
    }
};