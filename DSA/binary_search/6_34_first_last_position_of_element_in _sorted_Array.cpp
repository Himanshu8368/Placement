#include <vector>
using namespace std;
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int ans = -1;
        vector<int > solution;
        int low =0; int high = nums.size()-1;
        while(low<=high){
            int mid = (low+high)/2;
            if(nums[mid]<=target){
                if(nums[mid]==target){
                    ans = mid;
                }
                low= mid+1;
            }
            else{
                high=mid-1;
            }
        }
        int ans1 = ans;
        // solution.push_back(ans);
        ans =-1;
        low =0;high = nums.size()-1;
        while(low<=high){
            int mid = (low+high)/2;
            if(nums[mid]>=target){
                if(nums[mid]==target){
                    ans = mid;
                }
                high = mid-1;
            }
            else{
                low = mid+1;
            }
        }
        solution.push_back(ans);
        solution.push_back(ans1);
        return solution;
    }
};