import tkinter as tk
import pickle
from tkinter import font
from tkinter.messagebox import NO
from train_test import translate

'''
This python script create a GUI to visulize translation results.
'''

class GUI():
    def __init__(self,state):
        if(state == 0):
            self.main_window()
        else:
            self.translation_inter()

    def translation_inter(self):
        def translation(event=None):
            org_s = b1.get().strip()
            trans_s = translate(org_s)
            text_box.delete(1.0, 'end')
            text_box.insert('insert', trans_s)

        def reset():
            text_box.delete(1.0,'end')

        self.window = tk.Tk()
        self.window.title('Translation (EN->ZH)')
        self.window.geometry('900x600')

        # create canvas
        canvas = tk.Canvas(self.window, height=700, width=400)
        canvas.pack()

        # input test sentence
        L1 = tk.Label(self.window, text="Please input a sentence in English:", font=("Times New Roman", 12))
        L1.place(x=80, y=60)

        # text window
        b1 = tk.Entry(self.window, font=("Times New Roman", 12), show=None, width=85)
        b1.bind('<Return>', translation) 
        b1.place(x=80, y=100)

        # set translation function
        bt_click = tk.Button(self.window, text="Translate", width=25, height=2, command=translation, font=("Times New Roman", 10))
        bt_click.place(x=180, y=180)

        # set clearr function
        bt_trans = tk.Button(self.window, text="Clear the board",width=24,height=2,command=reset, font=("Times New Roman", 10))
        bt_trans.place(x=430, y=180)

        # set text box
        text_box = tk.Text(self.window, width=113, height=20, font=("Times New Roman", 10)) 
        text_box.place(x=80, y=260)

        # mainloop
        self.window.mainloop()

    def main_window(self):
        self.window=tk.Tk()
        self.window.title('Welcome to the translation system (EN->ZH)')
        self.window.geometry('450x300')
        # set canvas property
        self.canvas=tk.Canvas(self.window,height=300,width=500)
        self.canvas.pack(side='top')
        # set user information
        tk.Label(self.window,text='user:', font=("Times New Roman", 12)).place(x=100,y=100)
        tk.Label(self.window,text='pwd:',font=("Times New Roman", 12)).place(x=100,y=150)
        # input user name
        self.var_usr_name=tk.StringVar()
        self.entry_usr_name=tk.Entry(self.window,textvariable=self.var_usr_name, width=20)
        self.entry_usr_name.place(x=160,y=100)
        # input password
        self.var_usr_pwd=tk.StringVar()
        self.entry_usr_pwd=tk.Entry(self.window,textvariable=self.var_usr_pwd,show='*', width=20)
        self.entry_usr_pwd.place(x=160,y=150)
        # login or signin
        bt_login=tk.Button(self.window,text='Login',command=self.login, font=("Times New Roman", 10), width=10, height=2)
        bt_login.place(x=120,y=210)
        bt_logup=tk.Button(self.window,text='Register',command=self.register, font=("Times New Roman", 10), width=10, height=2)
        bt_logup.place(x=250,y=210)
        # bt_logquit=tk.Button(self.window,text='quit',command=quit)
        # bt_logquit.place(x=280,y=230)
        # main loop
        self.window.mainloop()

    # set login function
    def login(self):
        usr_name=self.var_usr_name.get()
        usr_pwd=self.var_usr_pwd.get()
        # retrive user information
        try:
            with open('usr_info.pickle','rb') as usr_file:
                usrs_info=pickle.load(usr_file)
        except FileNotFoundError:
            with open('usr_info.pickle','wb') as usr_file:
                usrs_info={'admin':'admin'}
                pickle.dump(usrs_info,usr_file)

        # tell if username and password are matched
        if usr_name in usrs_info:
            if usr_pwd == usrs_info[usr_name]:
                tk.messagebox.showinfo(title='welcome', message='login successfully!')
                # close login window
                self.window.destroy()
                gg=GUI(1)
                #self.translation_inter()
            else:
                tk.messagebox.showerror(message='password error!')
        # username and password cannot be empty
        elif usr_name=='':
            tk.messagebox.showerror('wrong','user_name is empty!')
        elif  usr_pwd=='':
            tk.messagebox.showerror('wrong','user_pwd is empty!')
        # pop not in register database
        else:
            is_signup=tk.messagebox.askyesno('welcome','You have not registered yet, register now?')
            if is_signup:
                self.register()

    # singin function
    def register(self):
        # get user information
        def getinfo():
            # get username and password
            reg_name=new_name.get()
            reg_pwd=new_pwd.get()
            reg_pwd2=new_pwd_confirm.get()

            # load user information
            try:
                with open('usr_info.pickle','rb') as usr_file:
                    exist_usr_info=pickle.load(usr_file)
            except FileNotFoundError:
                exist_usr_info={}

            # check if the username already exists, check if the password is correct
            if reg_name in exist_usr_info:
                tk.messagebox.showerror('wrong','user name already exists!')
            elif reg_pwd =='':
                tk.messagebox.showerror('wrong','username is empty!')
            elif reg_pwd2=='':
                tk.messagebox.showerror('wrong','password is empty')
            elif reg_pwd !=reg_pwd2:
                tk.messagebox.showerror('wrong','passwords do not match!')
            # store user information
            else:
                exist_usr_info[reg_name]=reg_pwd
                with open('usr_info.pickle','wb') as usr_file:
                    pickle.dump(exist_usr_info,usr_file)
                tk.messagebox.showinfo('welcome','register successfully')
                # close message box
                window_sign_up.destroy()
                
        # build user register window
        window_sign_up=tk.Toplevel(self.window)
        window_sign_up.geometry('350x200')
        window_sign_up.title('Register')
        # input username
        new_name=tk.StringVar()
        tk.Label(window_sign_up,text='User name: ', font=("Times New Roman", 12)).place(x=10,y=10)
        tk.Entry(window_sign_up,textvariable=new_name).place(x=150,y=10)
        # input password
        new_pwd=tk.StringVar()
        tk.Label(window_sign_up,text='Input pwd: ', font=("Times New Roman", 12)).place(x=10,y=50)
        tk.Entry(window_sign_up,textvariable=new_pwd,show='*').place(x=150,y=50)
        # repeat password
        new_pwd_confirm=tk.StringVar()
        tk.Label(window_sign_up,text='Input pwd agagin: ', font=("Times New Roman", 12)).place(x=10,y=90)
        tk.Entry(window_sign_up,textvariable=new_pwd_confirm,show='*').place(x=150,y=90)
        # register confirm
        bt_confirm_sign_up=tk.Button(window_sign_up,text='Register confirm',command=getinfo, font=("Times New Roman", 12))
        bt_confirm_sign_up.place(x=130,y=130)

    # qiut
    def quit(self):
        self.window.destroy() 

if __name__ == '__main__':
    gui=GUI(0)
