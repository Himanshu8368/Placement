#include <vector>
#include <set>  
#include <algorithm>
// its mostly same as 3 sum problem
// its mostoptimal solution is O(n^3) time complexity
// for other solution you can refer to 3 sum problem
using namespace std;
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> ans;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++){
            if(i!=0 && nums[i]==nums[i-1]) continue;
            for(int j=i+1;j<nums.size();j++){
                if(j!=i+1 && nums[j]==nums[j-1]) continue;
                long long new_target = (long long)target-nums[i]-nums[j];
                int low = j+1 ; int high = nums.size()-1;
                while(low<high){
                    long long sum = nums[low]+nums[high];
                    if(sum == new_target){
                        vector<int> temp ={nums[i],nums[j],nums[low],nums[high]};
                        ans.push_back(temp);
                        low++;
                        high--;
                        while(low<high && nums[low]== nums[low-1]) low++;
                        while(low<high && nums[high]== nums[high+1]) high--;
                    }
                    else if(sum>new_target) high--;
                    else low++;
                }
            }
        }
        return ans;
    }
};