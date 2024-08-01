css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20190710/ourlarge/pngtree-business-user-profile-vector-png-image_1541960.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

page_bg_image='''
<style>
.stApp{
background-image:url("https://th.bing.com/th/id/R.f73b1cd900192696cb3bc8e9c0effafc?rik=Xf1E25lWJe5r%2bg&riu=http%3a%2f%2fwallpapercave.com%2fwp%2fnZPfCfY.jpg&ehk=vCK8NxW7KKDEYcHsSU5VfcMRRzOOgpK9KN9Oum3dhuM%3d&risl=&pid=ImgRaw&r=0");
background-size: cover;
}
</style>
'''