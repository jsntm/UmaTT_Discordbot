# UmaTT_Discordbot
Umamusume team trials discord bot with OCR and data analysis

## Bot Host Usage (Windows)

```pip install -r requirements.txt```

Place your bot token inside umaTTauth.txt.

For best OCR accuracy, set an OpenAI API key before starting the bot:
```powershell $env:OPENAI_API_KEY = "your key"```

Fallbacks are EasyOCR and tesseract.

Start the bot as a background process:
```powershell python bot.py```

Optional environment variables:

- `DISCORD_GUILD_ID`: sync slash commands to one guild immediately while testing.
- `OPENAI_MODEL`: vision-capable model name used by the OpenAI OCR backend. Defaults to `gpt-4o-mini`.
- `UMA_BOT_DATA_DIR`: where CSV data is stored. Defaults to `data/`.
- `UMA_BOT_KEEP_IMAGES`: set to `1` to keep downloaded/cropped OCR images for debugging.
- `UMA_BOT_OCR_PROVIDER`: `auto`, `openai`, `easyocr`, or `tesseract`.
- `UMA_BOT_EASYOCR_GPU`: set to `1` to let EasyOCR use GPU acceleration when available.

## User Usage

<table>
  <tr>
    <td align="center"><b>TOP</b></td>
    <td align="center"><b>BOTTOM</b></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/5YYHwwn.png" width="200"></td>
    <td><img src="https://i.imgur.com/YTWa7OR.png" width="200"></td>
  </tr>
</table>

1. use `/team-replace` to fill out your team. Upon creation, each uma gets a 5 digit alphanumeric `uma_id`.
2. use `/change-ocr` to verify OCR works on top and bottom screenshots
3. use `/ocr` and attach up to 10 images. Multiple commands may be run at the same time with no issue.
4. use `/ocr2`, `/ocr3`, `/ocr4`, or `/ocr5` when uploading an exact number of trials. Every top/bottom attachment on these commands is mandatory, so Discord advances directly through the attachment fields while selecting images.

Use these slash commands to repair OCR mistakes:

- `/record-edit`: replace one record's score.
- `/record-add`: append one score for an uma currently in your team.
- `/record-delete`: preview an inclusive deletion range, then confirm or cancel with buttons.

Use these slash commands for data analysis:
- `/summary`: summarizes statistics for current team
- `/box-and-whisker`: self-explanatory
- `/box-and-whisker-custom`: compares comma-separated uma IDs, uma name, `current`, or `all`. Add optional track, `ace`/`nonace`, and running-style specifiers with dashes, for example `ab3de-front-nonace, current-dirt`. Set `merge-same-uma` to combine every matching configuration for each selector.

Use these slash commands for data export:
- `/get-current-team`
- `/get-records`
- `/get-all-umas`

Use these slash commands to modify your team:
- `/team-replace`: replace a spot in your team with a new uma
- `/team-swap`: given 2 `uma_id`, swap their positions on the team. 
- `/team edit`: given an `uma_id`, edit their information.

## Screenshot Stitching

`/stitch` combines two to ten same-sized screenshots of vertically scrolling content. Supply the images from top to bottom. Enable `debug` to include an alignment overlay for each successfully matched pair; alignment failures return the partial stitch and the image that failed.

`/change-stitch` shows or updates your per-user `crop_top`, `crop_bottom`, `window_height`, and `similarity_threshold` settings. Crop and window values are pixels, while similarity is a percentage from 0 to 100. Defaults are `0`, `0`, `50`, and `99` respectively.

More details on each slash command can be found in discord when using the bot.

## Data Layout

Runtime data is stored per Discord user:
```
data/
  ocr_settings.json
  users/
    <discord_user_id>/
      all_umas.csv
      current_team.csv
      records.csv
```

This folder can be copied with the bot to preserve data across restarts or machines.
