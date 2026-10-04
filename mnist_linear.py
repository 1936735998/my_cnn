import numpy as np
from tensorflow.keras.datasets import mnist


# =====================
# 数据
# =====================

(x_train,y_train),(x_test,y_test)=mnist.load_data()


x_train=x_train[:5000]
y_train=y_train[:5000]


x_train=x_train/255.0


# 展开图片

x_train=x_train.reshape(
    5000,
    784
)



# =====================
# 参数
# =====================

W=np.random.randn(
    784,
    10
)*0.01


b=np.zeros(10)



learning_rate=0.001



# =====================
# one-hot
# =====================

def one_hot(label):

    y=np.zeros(10)

    y[label]=1

    return y



# =====================
# 训练
# =====================

for epoch in range(10):


    total_loss=0


    for i in range(5000):


        # 输入

        x=x_train[i]


        target=one_hot(
            y_train[i]
        )



        # ---------
        # Forward
        # ---------

        prediction=np.dot(
            x,
            W
        )+b



        # Loss

        loss=np.mean(
            (prediction-target)**2
        )


        total_loss+=loss



        # ---------
        # Backward
        # ---------

        dY=2*(prediction-target)/10



        dW=np.dot(
            x.reshape(784,1),
            dY.reshape(1,10)
        )


        db=dY



        # ---------
        # Update
        # ---------

        W=W-learning_rate*dW

        b=b-learning_rate*db
        if i == 0:
            print(
                "loss:", loss,
                "W最大:", np.max(np.abs(W)),
                "梯度最大:", np.max(np.abs(dW))
            )



    print(
        "epoch:",
        epoch,
        "loss:",
        total_loss/5000
    )
    