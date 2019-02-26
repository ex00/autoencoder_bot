# autoencoder_bot

# Requirements
Python 3.7

## Run telegram bot
generate bot key via [BotFather](https://telegram.me/botfather), 
you can see full instruction for create bot [here](https://core.telegram.org/bots/faq)

### Run in docker container
save your bot key in file with name `env.list`. e.g
```bash
BOT_KEY=000000:YYYYYEEEEEWWWWWWWWWWQQQQQQQQ
```
Build and run container with bot
```bash
git clone https://github.com/ex00/autoencoder_bot.git
cd autoencoder_bot
docker build -t autoencoder_bot .
docker run --env-file /path/to/your/env.list --rm autoencoder_bot python /bot/bot.py
```

### Run locally
```bash
git clone https://github.com/ex00/autoencoder_bot.git
cd autoencoder_bot
set BOT_KEY=00000:YYYYYEEEEEWWWWWWWWWWQQQQQQQQ # it's your bot key
set MODEL=$(pwd)/autoencoder/saved/model.torch # path to saved model
pip3 install -r requirements.txt
python3 /bot/bot.py
```
