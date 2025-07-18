#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
// Problem: 931. Minimum Falling Path Sum
// Link: https://leetcode.com/problems/minimum-falling-path-sum/
// memorization approach give tle on test case 49/52
class Solution {
public:
    int solve(int i ,int j ,int n, vector<vector<int>>& matrix,vector<vector<int>> &dp){
        if(j<0 || j>=n) return INT_MAX;
        if(i==n-1) return matrix[i][j];
        if(dp[i][j]!=-1) return dp[i][j];
        int up = solve(i+1,j-1,n,matrix,dp);
        int left_diagonal = solve(i+1,j,n,matrix,dp);
        int right_diagonal = solve(i+1,j+1,n,matrix,dp);

        return dp[i][j] = matrix[i][j]+min(up,min(left_diagonal,right_diagonal));

    }
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size();
        vector<vector<int>> dp (n,vector<int>(n,-1));
        int maxi = INT_MAX;
        for(int i=0;i<n;i++){
            int temp = solve(0,i,n,matrix,dp);
            maxi = min(maxi,temp);
        }
        return maxi;
    }
};
// Tabulation approach
// This approach is better than memorization approach
class Solution1 {
public:
    int solve(int n, vector<vector<int>>& matrix,vector<vector<int>> &dp){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(i==0) {
                    dp[i][j] = matrix[i][j];
                    continue;
                }
                int up = dp[i-1][j];
                int left = INT_MAX;
                int right = INT_MAX;
                if(j>0){
                    left = dp[i-1][j-1];
                }
                if(j<n-1){
                    right = dp[i-1][j+1];
                }
                dp[i][j]= matrix[i][j]+min({up,left,right});
 
            }
        }
        int mini = INT_MAX;
        for(int i=0;i<n;i++){
            mini = min(mini,dp[n-1][i]);
        }
        return mini;

    }
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size();
        vector<vector<int>> dp (n,vector<int>(n,-1));

        return solve(n,matrix,dp);
    }
};



