#include <vector>
#include <string>
#include <algorithm>
using namespace std;
class Solution {
public:
    void solve(int i, int sum , vector<vector<int>>& ans,vector<int>& nums, int target,vector<int>& temp){
        if(i>=nums.size()){
            if(sum == target){
                ans.push_back(temp);
            }
            return ;
        }
        if(sum + nums[i]<= target){   // Only consider the current number if it does not exceed the target
            sum = sum +nums[i];
            temp.push_back(nums[i]);
            solve(i,sum,ans,nums,target,temp);
            temp.pop_back();
            sum = sum - nums[i];
        }
        solve(i+1,sum,ans,nums,target,temp);
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        int sum =0;
        vector<vector<int>> ans;
        vector<int> temp;
        // for this problem , sorting the candidates is not necessary, but it helps to prune the search space
        // as we can skip the candidates which are greater than the target
        //  it run fine without sorting as well
        sort(candidates.begin(),candidates.end()); // it helps to prune the search space
        solve(0,sum,ans,candidates,target , temp);
        return ans;
    }
};

// modified and optimized version of the code
class Solution1 {
public:
    void solve(int i , int sum , int target ,vector<int>& nums ,vector<vector<int>> & ans,vector<int> & temp){
        if(i>= nums.size()){
            if(sum == target){
                ans.push_back(temp);
            }
            return;
        }
        if(sum + nums[i]<= target){
            sum = sum+nums[i];
            temp.push_back(nums[i]);
            solve(i,sum,target , nums ,ans ,temp);
            sum = sum- nums[i];
            temp.pop_back();
            while(i<=nums.size()){
                if(nums[i]== nums[i+1]) i++;
                else{
                    break;
                }
            }
        }
        solve(i+1,sum,target , nums ,ans ,temp);
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<vector<int>> ans;
        vector<int> temp;
        int sum =0;
        solve(0,sum,target , candidates ,ans ,temp);
        return ans;
    }
};


