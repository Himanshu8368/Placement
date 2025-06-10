#include <vector>
#include <stack>
using namespace std;
struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 };

class Solution {
public:
    // void solve(TreeNode* root , vector<int> & arr){
    //     if(root == nullptr) return ;
    //     arr.push_back(root->val);
    //     solve(root->left,arr);
    //     solve(root->right,arr);
    // }
    void solve(TreeNode* root , vector<int> & arr){
        if(root == nullptr) return ;
        stack<TreeNode*> st;
        st.push(root);
        while(!st.empty()){
            TreeNode * node = st.top();
            st.pop();
            arr.push_back(node->val);
            if(node->right != nullptr)  st.push(node->right);
            if(node->left != nullptr)   st.push(node->left);
            
        }

    }
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> arr;
        solve(root,arr);
        return arr;
    }
};