#include <vector>
using namespace std;
class Solution {
public:
    vector<int> getFloorAndCeil(vector<int> nums, int x) {
        int ans = nums.size();
        vector<int>solution;
        int low =0;int high = nums.size()-1;
        while(low<=high){
            int mid = (low +high)/2;
            if(nums[mid]<=x){
                ans = mid;
                low = mid+1;
            }
            else{
                high = mid-1;
            }
        }
        solution.push_back(nums[ans]);
        ans = nums.size();
        low =0;high = nums.size()-1;
        while(low<=high){
            int mid = (low+high)/2;
            if(nums[mid]>=x){
                ans = mid;
                high = mid-1;
            }
            else{
                low = mid+1;
            }
        }
        solution.push_back(nums[ans]);
        return solution;
    }
};