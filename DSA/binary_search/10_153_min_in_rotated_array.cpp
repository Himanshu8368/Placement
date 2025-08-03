// 153. Find Minimum in Rotated Sorted Array
// https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
// Time Complexity: O(log n)
// Space Complexity: O(1)
#include <vector>
#include <algorithm>
using namespace std;
class Solution {
public:
    int findMin(vector<int>& nums) {
        int low =0;
        int high = nums.size()-1;
        int ans =INT_MAX;
        while(low<high){
            int mid = (low+high)/2;
            ans = min(ans,nums[mid]);
            if(nums[mid]>nums[high]){
                low = mid+1;
            }
            else{
                high = mid-1;
            }
        }
        ans = min(ans,nums[low]);
        return ans;
    }
};