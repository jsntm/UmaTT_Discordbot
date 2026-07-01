from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands

from ttbot import config
from ttbot.constants import ORDER_KEYS, SORT_KEYS, STRATEGIES, TRACKS
from ttbot.names import NameMatcher
from ttbot.ocr import OCRFailure, OCRService
from ttbot.records import add_manual_record, delete_record_range, edit_record_score, preview_delete_records
from ttbot.reporting import build_boxplot, build_records_export, build_summary_rows, format_summary_table, write_all_umas_csv, write_records_csv
from ttbot.storage import UserStore
from ttbot.team import (
    OCRAddResult,
    TeamError,
    add_records_from_ocr,
    format_uma_id,
    ensure_full_team,
    format_added_uma,
    format_current_team,
    format_removed_uma,
    format_team_entry,
    replace_team_slot,
    swap_team_members,
    update_uma,
)
from ttbot.validation import normalize_uma_id


class UmaBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.name_matcher = NameMatcher.from_reference_file(config.REFERENCE_NAMES_FILE)
        self.ocr_service = OCRService(self.name_matcher)

    async def setup_hook(self) -> None:
        guild_id = os.environ.get("DISCORD_GUILD_ID")
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            print(f"Synced {len(synced)} commands to guild {guild_id}")
        else:
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} global commands")


bot = UmaBot()


def store_for(user: discord.abc.User) -> UserStore:
    return UserStore(str(user.id))


def upload_limit(interaction: discord.Interaction) -> int:
    if interaction.guild and getattr(interaction.guild, "filesize_limit", None):
        return int(interaction.guild.filesize_limit)
    return config.MAX_DISCORD_FILE_BYTES


async def respond(
    interaction: discord.Interaction,
    content: str,
    *,
    file: Optional[discord.File] = None,
    view: Optional[discord.ui.View] = None,
) -> None:
    content = content[:1900] if len(content) > 1900 else content
    kwargs: dict[str, object] = {"content": content}
    if file is not None:
        kwargs["file"] = file
    if view is not None:
        kwargs["view"] = view
    try:
        if interaction.response.is_done():
            await interaction.followup.send(**kwargs)
        else:
            await interaction.response.send_message(**kwargs)
    except discord.NotFound:
        print(f"Could not respond to expired interaction: {content[:200]}")


async def respond_files(
    interaction: discord.Interaction,
    content: str,
    *,
    files: list[discord.File],
    view: Optional[discord.ui.View] = None,
) -> None:
    content = content[:1900] if len(content) > 1900 else content
    kwargs: dict[str, object] = {"content": content}
    if files:
        kwargs["files"] = files
    if view is not None:
        kwargs["view"] = view
    try:
        if interaction.response.is_done():
            await interaction.followup.send(**kwargs)
        else:
            await interaction.response.send_message(**kwargs)
    except discord.NotFound:
        print(f"Could not respond to expired interaction: {content[:200]}")


def target_user_or_sender(interaction: discord.Interaction, user: Optional[discord.User]) -> discord.abc.User:
    return user or interaction.user


def user_label(user: discord.abc.User) -> str:
    name = getattr(user, "display_name", None) or getattr(user, "global_name", None) or user.name
    safe_name = str(name).replace("`", "'")
    return f"`@{safe_name}`"


def make_choices(values: list[str]) -> list[app_commands.Choice[str]]:
    return [app_commands.Choice(name=value, value=value) for value in values]


track_choices = make_choices(TRACKS)
strategy_choices = make_choices(STRATEGIES)
sort_choices = make_choices(SORT_KEYS)
order_choices = make_choices(ORDER_KEYS)
screenshot_choices = make_choices(["top", "bottom"])


def raw_ocr_block(raw_text: str, *, limit: int = 900) -> str:
    raw = raw_text.strip() or "(no OCR text returned)"
    if len(raw) > limit:
        raw = raw[:limit] + "\n..."
    return f"Raw OCR output:\n```text\n{raw}\n```"


def format_ocr_added_records(result: OCRAddResult) -> str:
    display_rows = [
        {"ind": str(row.index), "uma": row.entry.name, "pts": f"{row.score:,}"}
        for row in result.added
    ]
    lines = []
    if display_rows:
        columns = [("ind", "ind"), ("uma", "uma"), ("pts", "pts")]
        widths = {
            key: max(len(label), *(len(record[key]) for record in display_rows))
            for key, label in columns
        }

        def render(record: dict[str, str] | None = None) -> str:
            values = {key: label if record is None else record[key] for key, label in columns}
            return " | ".join(values[key].ljust(widths[key]) for key, _ in columns)

        separator = "-+-".join("-" * widths[key] for key, _ in columns)
        lines.append("```text\n" + "\n".join([render(), separator, *(render(row) for row in display_rows)]) + "\n```")
    if result.warnings:
        lines.append("Warnings:\n" + "\n".join(f"- {warning}" for warning in result.warnings))
    return "\n".join(lines)


def format_change_ocr_message(result) -> str:
    x1, y1, x2, y2 = result.region
    return (
        f"{bot.ocr_service.format_rows(result.rows)}\n\n"
        f"OCR region top-left ({x1}, {y1}), bottom-right ({x2}, {y2})\n"
        f"{raw_ocr_block(result.raw_text)}"
    )


class RecordDeleteConfirmView(discord.ui.View):
    def __init__(self, user_id: str, start_index: int, end_index: int | None) -> None:
        super().__init__(timeout=300)
        self.user_id = str(user_id)
        self.start_index = start_index
        self.end_index = end_index

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if str(interaction.user.id) == self.user_id:
            return True
        await interaction.response.send_message("Only the user who ran `/record-delete` can use these buttons.", ephemeral=True)
        return False

    @discord.ui.button(label="CONFIRM", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        try:
            delete_record_range(UserStore(self.user_id), self.start_index, self.end_index)
        except TeamError as exc:
            await interaction.response.edit_message(content=f"Could not delete records: {exc}", attachments=[], view=None)
            self.stop()
            return
        await interaction.response.edit_message(content="records deleted successfully", attachments=[], view=None)
        self.stop()

    @discord.ui.button(label="CANCEL", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="canceled records deletion", attachments=[], view=None)
        self.stop()


@bot.tree.command(name="team-replace", description="Add an uma to a team slot, replacing the current runner if needed.")
@app_commands.choices(track=track_choices, strategy=strategy_choices)
@app_commands.describe(
    track="sprint, mile, medium, long, or dirt",
    position="1 is ace; 2 and 3 are the other runners",
    strategy="front, pace, late, or end",
    outfit="Outfit/style label, such as new years or festival",
    name="Uma name. Minor typos and common abbreviations are accepted.",
    rating="Positive integer rating",
    date_acquired="MM/DD/YYYY or MM-DD-YYYY",
)
async def team_replace(
    interaction: discord.Interaction,
    track: str,
    position: int,
    strategy: str,
    outfit: str,
    name: str,
    rating: int,
    date_acquired: str,
) -> None:
    try:
        store = store_for(interaction.user)
        result = replace_team_slot(
            store,
            bot.name_matcher,
            track,
            position,
            strategy,
            outfit,
            name,
            rating,
            date_acquired,
        )
        lines = [format_added_uma(result.added)]
        if result.removed:
            lines.append(format_removed_uma(result.removed, result.removed_record_count))
        await respond(interaction, "\n".join(lines))
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="team-swap", description="Swap two team members, or bring a benched uma back into a team slot.")
@app_commands.describe(uma_1_id="Five-character uma code", uma_2_id="Five-character uma code")
async def team_swap(interaction: discord.Interaction, uma_1_id: str, uma_2_id: str) -> None:
    try:
        result = swap_team_members(store_for(interaction.user), normalize_uma_id(uma_1_id), normalize_uma_id(uma_2_id))
        await respond(interaction, "\n".join(result.messages))
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="team-edit", description="Edit an uma's saved details or current team slot.")
@app_commands.choices(track=track_choices, strategy=strategy_choices)
@app_commands.describe(
    uma_id="Five-character uma code",
    track="Optional new track",
    position="Optional new team position",
    strategy="Optional new strategy",
    outfit="Optional new outfit/style label",
    name="Optional new uma name",
    rating="Optional new rating",
    date_acquired="Optional new acquisition date",
)
async def team_edit(
    interaction: discord.Interaction,
    uma_id: str,
    track: Optional[str] = None,
    position: Optional[int] = None,
    strategy: Optional[str] = None,
    outfit: Optional[str] = None,
    name: Optional[str] = None,
    rating: Optional[int] = None,
    date_acquired: Optional[str] = None,
) -> None:
    try:
        old_entry, new_entry = update_uma(
            store_for(interaction.user),
            bot.name_matcher,
            normalize_uma_id(uma_id),
            track,
            position,
            strategy,
            outfit,
            name,
            rating,
            date_acquired,
        )
        await respond(interaction, f"{format_team_entry(old_entry)}\nhas been updated to\n{format_team_entry(new_entry)}")
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="ocr", description="Read a top and bottom score screenshot and append records.")
@app_commands.describe(top_image="Top screenshot", bottom_image="Bottom screenshot")
async def ocr(interaction: discord.Interaction, top_image: discord.Attachment, bottom_image: discord.Attachment) -> None:
    await interaction.response.defer(thinking=True)
    temp_paths: list[Path] = []
    try:
        store = store_for(interaction.user)
        ensure_full_team(store)
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            tmp = Path(tmp_name)
            top_path = tmp / f"top{Path(top_image.filename).suffix or '.jpg'}"
            bottom_path = tmp / f"bottom{Path(bottom_image.filename).suffix or '.jpg'}"
            await top_image.save(top_path)
            await bottom_image.save(bottom_path)
            temp_paths.extend([top_path, bottom_path])

            top_result = await asyncio.to_thread(bot.ocr_service.process_image, store.user_id, top_path, "top", tmp)
            bottom_result = await asyncio.to_thread(bot.ocr_service.process_image, store.user_id, bottom_path, "bottom", tmp)
            rows = bot.ocr_service.merge_rows([top_result.rows, bottom_result.rows])
            result = add_records_from_ocr(store, rows, interaction.created_at)
            await respond(interaction, format_ocr_added_records(result))
    except OCRFailure as exc:
        await respond(interaction, exc.to_user_message())
    except TeamError as exc:
        await respond(interaction, str(exc))
    finally:
        if not config.KEEP_IMAGES:
            for path in temp_paths:
                path.unlink(missing_ok=True)


@bot.tree.command(name="record-edit", description="Edit the score for one record by its records.csv row index.")
@app_commands.describe(record_index="1-based index from /get-records or /ocr output", score="Replacement positive integer score")
async def record_edit(interaction: discord.Interaction, record_index: int, score: int) -> None:
    try:
        result = edit_record_score(store_for(interaction.user), record_index, score)
        await respond(
            interaction,
            f"score at index {result.index} for {result.name} ({result.outfit}) has been changed from `{result.old_score}` to `{result.new_score}`",
        )
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="record-add", description="Manually append one record for a current team uma.")
@app_commands.rename(record_datetime="datetime")
@app_commands.describe(
    uma_id="Five-character uma code currently in your team",
    record_datetime="MM/DD/YYYY or MM-DD-YYYY; stored at 00:00:00",
    score="Positive integer score",
)
async def record_add(interaction: discord.Interaction, uma_id: str, record_datetime: str, score: int) -> None:
    try:
        result = add_manual_record(store_for(interaction.user), uma_id, record_datetime, score)
        await respond(
            interaction,
            (
                f"appended index {result.index}: {format_uma_id(result.uma_id)}, {result.name} ({result.outfit}) "
                f"rating {result.rating}, acquired {result.date_acquired} running as {result.strategy} "
                f"{result.track} {result.position} to records with {result.display_datetime}, {result.score}"
            ),
        )
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="record-delete", description="Delete one record or an inclusive range after confirmation.")
@app_commands.describe(start_index="First 1-based record index to delete", end_index="Optional final 1-based record index to delete")
async def record_delete(interaction: discord.Interaction, start_index: int, end_index: Optional[int] = None) -> None:
    await interaction.response.defer(thinking=True)
    try:
        store = store_for(interaction.user)
        rows = preview_delete_records(store, start_index, end_index)
        view = RecordDeleteConfirmView(str(interaction.user.id), start_index, end_index)
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            path = Path(tmp_name) / "to-be-deleted.csv"
            write_records_csv(path, rows)
            if path.stat().st_size > upload_limit(interaction):
                await respond(interaction, "the to-be-deleted-csv is too large to send. Do you want to proceed with deletion?", view=view)
                return
            await respond(
                interaction,
                "These rows are marked to be deleted. Proceed?",
                file=discord.File(path, filename="to-be-deleted.csv"),
                view=view,
            )
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="change-ocr", description="Preview or update OCR crop settings for one screenshot.")
@app_commands.choices(screenshot_type=screenshot_choices)
@app_commands.describe(
    image="Screenshot to test",
    screenshot_type="top or bottom",
    top_left_x="Optional crop top-left x",
    top_left_y="Optional crop top-left y",
    bottom_right_x="Optional crop bottom-right x",
    bottom_right_y="Optional crop bottom-right y",
)
async def change_ocr(
    interaction: discord.Interaction,
    image: discord.Attachment,
    screenshot_type: str,
    top_left_x: Optional[int] = None,
    top_left_y: Optional[int] = None,
    bottom_right_x: Optional[int] = None,
    bottom_right_y: Optional[int] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    provided = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    if any(value is not None for value in provided) and any(value is None for value in provided):
        await respond(interaction, "Provide all four crop coordinates: top_left_x, top_left_y, bottom_right_x, bottom_right_y.")
        return

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
        tmp = Path(tmp_name)
        image_path = tmp / f"ocr{Path(image.filename).suffix or '.jpg'}"
        await image.save(image_path)
        try:
            coords = None
            if all(value is not None for value in provided):
                coords = ((int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)))
            result = await asyncio.to_thread(
                bot.ocr_service.process_image,
                str(interaction.user.id),
                image_path,
                screenshot_type,
                tmp,
                update_coords=coords,
            )
            message = format_change_ocr_message(result)
        except OCRFailure as exc:
            message = exc.to_user_message()
            result = exc.result

        files = []
        if result and result.highlight_path and result.highlight_path.exists():
            files.append(discord.File(result.highlight_path, filename="ocr-region.png"))
        await respond_files(interaction, message, files=files)


@bot.tree.command(name="get-records", description="Export score records as a CSV.")
@app_commands.describe(user="Discord user, defaults to you", scope="current, all, or a five-character uma code")
async def get_records(interaction: discord.Interaction, scope: str = "current", user: Optional[discord.User] = None) -> None:
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        store = store_for(target)
        rows = build_records_export(store, scope)
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            path = Path(tmp_name) / "records.csv"
            write_records_csv(path, rows)
            if path.stat().st_size > upload_limit(interaction):
                await respond(interaction, f"Records for {target_label}\nThat CSV is too large for a Discord message. Ask the bot host to retrieve it directly from the data folder.")
                return
            await respond(interaction, f"Records for {target_label}", file=discord.File(path, filename="records.csv"))
    except TeamError as exc:
        await respond(interaction, f"Records for {target_label}\n{exc}")


@bot.tree.command(name="get-current-team", description="Show the current team.")
@app_commands.describe(user="Discord user, defaults to you")
async def get_current_team(interaction: discord.Interaction, user: Optional[discord.User] = None) -> None:
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        await respond(interaction, f"Current team for {target_label}\n{format_current_team(store_for(target))}")
    except TeamError as exc:
        await respond(interaction, f"Current team for {target_label}\n{exc}")


@bot.tree.command(name="get-all-umas", description="Export all saved umas as a CSV.")
@app_commands.describe(user="Discord user, defaults to you")
async def get_all_umas(interaction: discord.Interaction, user: Optional[discord.User] = None) -> None:
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        store = store_for(target)
        rows = store.read_all_umas()
        if not rows:
            raise TeamError("That user does not have any umas saved.")
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            path = Path(tmp_name) / "all_umas.csv"
            write_all_umas_csv(path, rows)
            if path.stat().st_size > upload_limit(interaction):
                await respond(interaction, f"All umas for {target_label}\nThat CSV is too large for a Discord message. Ask the bot host to retrieve it directly from the data folder.")
                return
            await respond(interaction, f"All umas for {target_label}", file=discord.File(path, filename="all_umas.csv"))
    except TeamError as exc:
        await respond(interaction, f"All umas for {target_label}\n{exc}")


@bot.tree.command(name="summary", description="Show score stats for the current team.")
@app_commands.choices(sort=sort_choices, order=order_choices)
@app_commands.describe(user="Discord user, defaults to you", sort="Sort order", order="ascending or descending")
async def summary(
    interaction: discord.Interaction,
    sort: str = "track",
    order: str = "descending",
    user: Optional[discord.User] = None,
) -> None:
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        rows = build_summary_rows(store_for(target), sort, order)
        if not rows:
            raise TeamError("That user does not have matching records for the current team.")
        await respond(interaction, f"Summary for {target_label}\n{format_summary_table(rows)}")
    except TeamError as exc:
        await respond(interaction, f"Summary for {target_label}\n{exc}")


@bot.tree.command(name="box-and-whisker", description="Plot current team score distributions.")
@app_commands.choices(sort=sort_choices, order=order_choices)
@app_commands.describe(user="Discord user, defaults to you", sort="Sort metric", order="ascending or descending")
async def box_and_whisker(
    interaction: discord.Interaction,
    sort: str = "median",
    order: str = "descending",
    user: Optional[discord.User] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            path = Path(tmp_name) / "box-and-whisker.png"
            build_boxplot(store_for(target), path, sort, order)
            if path.stat().st_size > upload_limit(interaction):
                await respond(interaction, f"Box and whisker for {target_label}\nThat plot is too large for a Discord message.")
                return
            await respond(interaction, f"Box and whisker for {target_label}", file=discord.File(path, filename="box-and-whisker.png"))
    except TeamError as exc:
        await respond(interaction, f"Box and whisker for {target_label}\n{exc}")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    if isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, (TeamError, ValueError)):
        await respond(interaction, str(error.original))
        return
    await respond(interaction, f"Something went wrong while running that command: {error}")


def main() -> None:
    token = config.read_discord_token()
    bot.run(token)


if __name__ == "__main__":
    main()
