#include <vector>
using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int ans = -1;
        int low =0; int high = nums.size()-1;
        while(low<=high){
            int mid = (low+high)/2;
            if(nums[mid]== target) {
                ans = mid;
                return ans;
            }
            else if (nums[high]>=nums[mid]){
                if(nums[mid]<target && nums[high]>=target){                    
                    low = mid+1;
                }
                else{
                    high = mid-1;
                }
            }
            else if(nums[mid]>=nums[low]){
                if(nums[mid]>target && nums[low]<=target){                    
                    high = mid -1;
                }
                else{
                    low = mid+1;
                }
            }
        }
        return ans ;
    }
};