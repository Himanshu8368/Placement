#include <vector>
using namespace std;
class Solution {
public:
    void solve(vector<int>& nums,vector<vector<int>>& ans ,int  i ,int n ,vector<int>& temp ){
        if (i>= n) {
            ans.push_back(temp);
            return;
        }
        temp.push_back(nums[i]);
        // ans.push_back(temp);
        solve(nums,ans , i+1 ,nums.size() ,temp);
        temp.pop_back();
        // ans.push_back(temp);
        solve(nums,ans , i+1 ,nums.size() ,temp);
        
        
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans ;
        vector<int> temp;
        solve(nums,ans , 0 ,nums.size() ,temp);
        return ans;
    }
};