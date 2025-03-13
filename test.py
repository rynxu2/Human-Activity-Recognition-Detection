import requests

# Thay YOUR_APP_TOKEN và YOUR_USER_KEY bằng thông tin của bạn
app_token = "a8kvr5dh5rtnz16tw9pm6kqo75wtkc"
user_key = "u2wdn43qm1ed5mwguhs4wftecin7pw"
message = "Thông báo từ Python!"

# Gửi yêu cầu đến API của Pushover
response = requests.post("https://api.pushover.net/1/messages.json", data={
    "token": app_token,
    "user": user_key,
    "message": message,
    "title": "Cảnh báo hệ thống",
    "priority": 2,
    "retry": 60,
    "expire": 3600,
    "sound": "pushover"
})

# Kiểm tra phản hồi từ server
if response.status_code == 200:
    print("Thông báo đã được gửi thành công!")
else:
    print("Lỗi khi gửi thông báo:", response.text)
