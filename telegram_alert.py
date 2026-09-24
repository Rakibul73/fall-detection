import requests


def send_telegram_alert(bot_token, chat_id, message, image_path=None):
    try:
        if image_path:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, "rb") as photo:
                requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": message},
                    files={"photo": photo},
                    timeout=10
                )
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        print(f"[telegram] failed to send alert: {e}")
