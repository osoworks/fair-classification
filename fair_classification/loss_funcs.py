import numpy as np

def squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def logistic_loss(y_true, y_pred):
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))

if __name__ == "__main__":
    # 예제 데이터 생성
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0.9, 0.2, 0.8, 0.6])
    
    # 손실 함수 계산
    sq_loss = squared_loss(y_true, y_pred)
    log_loss = logistic_loss(y_true, y_pred)
    
    # 결과 출력
    print("Squared Loss:", sq_loss)
    print("Logistic Loss:", log_loss)