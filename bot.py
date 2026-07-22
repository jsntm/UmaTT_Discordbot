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
from ttbot.image_uploads import PreparedImage, prepare_image_attachment
from ttbot.names import NameMatcher
from ttbot.ocr import OCRFailure, OCRService
from ttbot.reference_data import download_missing_thumbnails, generate_reference_files
from ttbot.records import add_manual_record, delete_record_range, edit_record_score, preview_delete_records
from ttbot.reporting import (
    build_boxplot,
    build_custom_boxplot,
    build_records_export,
    build_summary_rows,
    format_summary_table,
    write_all_umas_csv,
    write_records_csv,
)
from ttbot.storage import UserStore, get_stitch_setting, set_stitch_setting
from ttbot.stitching import StitchingError, parse_stitch_settings, stitch_image_sequence
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
    get_current_team_entries,
    ocr_bijection_issues,
    replace_team_slot,
    swap_team_members,
    update_uma,
)
from ttbot.validation import normalize_uma_id


REFERENCE_ROWS = generate_reference_files()


class UmaBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.name_matcher = NameMatcher.from_reference_files(
            config.UMA_NAMES_FILE,
            config.UMA_NAME_ALIASES_FILE,
            config.OUTFIT_NAMES_FILE,
            config.OUTFIT_NAME_ALIASES_FILE,
            config.UMA_THUMBS_DIR,
        )
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
    interaction_limit = getattr(interaction, "filesize_limit", None)
    if interaction_limit:
        return int(interaction_limit)
    if interaction.guild and getattr(interaction.guild, "filesize_limit", None):
        return int(interaction.guild.filesize_limit)
    return config.MAX_DISCORD_FILE_BYTES


async def prepare_generated_image_files(
    interaction: discord.Interaction,
    paths: list[Path],
) -> tuple[list[discord.File], list[PreparedImage]]:
    limit = upload_limit(interaction)
    prepared = [await asyncio.to_thread(prepare_image_attachment, path, limit) for path in paths]
    files = [discord.File(item.path, filename=item.path.name) for item in prepared]
    return files, prepared


def image_compression_note(prepared: list[PreparedImage]) -> str:
    count = sum(item.compressed for item in prepared)
    if not count:
        return ""
    noun = "image" if count == 1 else "images"
    return f"\nCompressed {count} generated {noun} to fit Discord's upload limit."


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


def format_ocr_trial_tables(team_entries, trial_results: list[OCRAddResult | None], *, max_chars: int = 1800) -> list[str]:
    ordered_ids: list[str] = []
    entries_by_id = {entry.uma_id.lower(): entry for entry in team_entries}
    for result in trial_results:
        if result is None:
            continue
        for added in result.added:
            uma_id = added.entry.uma_id.lower()
            if uma_id not in ordered_ids:
                ordered_ids.append(uma_id)
    ordered_ids.extend(uma_id for uma_id in entries_by_id if uma_id not in ordered_ids)

    trial_maps = [
        {} if result is None else {added.entry.uma_id.lower(): added for added in result.added}
        for result in trial_results
    ]
    trial_widths = []
    for trial_index, records in enumerate(trial_maps, start=1):
        values = [f"trial {trial_index}"]
        values.extend(str(record.index) for record in records.values())
        values.extend(f"{record.score:,}" for record in records.values())
        trial_widths.append(max(len(value) for value in values))

    def render_values(label: str, values: list[str]) -> str:
        return " | ".join([label.ljust(3), *(value.ljust(width) for value, width in zip(values, trial_widths))])

    header = render_values("uma", [f"trial {index}" for index in range(1, len(trial_maps) + 1)])
    table_width = max(len(header), *(len(entries_by_id[uma_id].name) for uma_id in ordered_ids))
    header = header.ljust(table_width)
    blocks = []
    for uma_id in ordered_ids:
        entry = entries_by_id[uma_id]
        indexes = [str(records[uma_id].index) if uma_id in records else "" for records in trial_maps]
        scores = [f"{records[uma_id].score:,}" if uma_id in records else "" for records in trial_maps]
        name_line = entry.name + "-" * (table_width - len(entry.name))
        blocks.append("\n".join([name_line, render_values("ind", indexes).ljust(table_width), render_values("pts", scores).ljust(table_width)]))

    chunks: list[str] = []
    current_blocks: list[str] = []
    for block in blocks:
        candidate = "```text\n" + "\n".join([header, *current_blocks, block]) + "\n```"
        if current_blocks and len(candidate) > max_chars:
            chunks.append("```text\n" + "\n".join([header, *current_blocks]) + "\n```")
            current_blocks = [block]
        else:
            current_blocks.append(block)
    if current_blocks:
        chunks.append("```text\n" + "\n".join([header, *current_blocks]) + "\n```")
    return chunks


def format_bullet_messages(header: str, messages: list[str], *, max_chars: int = 1800) -> list[str]:
    chunks = []
    current = header
    for message in messages:
        bullet = f"- {message}"
        candidate = current + "\n" + bullet
        if current != header and len(candidate) > max_chars:
            chunks.append(current)
            current = header + "\n" + bullet
        else:
            current = candidate
    if current != header:
        chunks.append(current)
    return chunks


def format_stitch_settings(settings) -> str:
    threshold = f"{settings.similarity_threshold:g}%"
    rows = [
        ("crop_top", f"{settings.crop_top} px"),
        ("crop_bottom", f"{settings.crop_bottom} px"),
        ("window_height", f"{settings.window_height} px"),
        ("similarity_threshold", threshold),
    ]
    width = max(len(label) for label, _ in rows)
    return "```text\n" + "\n".join(f"{label.ljust(width)} | {value}" for label, value in rows) + "\n```"


def format_change_ocr_message(result) -> str:
    x1, y1, x2, y2 = result.region
    return (
        f"{result.region_mode.title()} parsed OCR output:\n"
        f"{bot.ocr_service.format_rows(result.rows)}\n\n"
        f"{result.region_mode.title()} OCR region top-left ({x1}, {y1}), bottom-right ({x2}, {y2})\n"
        f"{raw_ocr_block(result.raw_text)}"
    )


async def prepare_change_ocr_response(
    interaction: discord.Interaction,
    user_id: str,
    image_path: Path,
    screenshot_type: str,
    work_dir: Path,
    *,
    manual: bool,
    update_coords: tuple[int | None, int | None, int | None, int | None] | None = None,
) -> tuple[str, list[discord.File]]:
    result = None
    try:
        result = await asyncio.to_thread(
            bot.ocr_service.process_image,
            user_id,
            image_path,
            screenshot_type,
            work_dir,
            update_coords=update_coords,
            manual=manual,
        )
        message = format_change_ocr_message(result)
    except OCRFailure as exc:
        message = exc.to_user_message()
        result = exc.result

    preview_paths = []
    if result and result.highlight_path and result.highlight_path.exists():
        preview_paths.append(result.highlight_path)
    files, prepared = await prepare_generated_image_files(interaction, preview_paths)
    return message + image_compression_note(prepared), files


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
    outfit="Uma outfit name; minor typos and configured aliases are accepted",
    name="Uma name. Minor typos and common abbreviations are accepted.",
    rating="Positive integer rating",
    date_acquired="MM/DD/YYYY, MM-DD-YYYY, or today (UTC)",
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
    outfit="Optional official or fuzzy-matched outfit name",
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


async def _run_ocr_trials(
    interaction: discord.Interaction,
    image_pairs: list[tuple[discord.Attachment, discord.Attachment]],
    *,
    manual: bool = False,
) -> None:
    await interaction.response.defer(thinking=True)
    try:
        store = store_for(interaction.user)
        ensure_full_team(store)
        team_entries = get_current_team_entries(store)
        candidate_names = [entry.name for entry in team_entries]
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            tmp = Path(tmp_name)
            trial_results: list[OCRAddResult | None] = []
            trial_warnings: list[str] = []
            failure_details: list[str] = []
            for trial_index, (top, bottom) in enumerate(image_pairs, start=1):
                top_path = tmp / f"trial-{trial_index}-top{Path(top.filename).suffix or '.jpg'}"
                bottom_path = tmp / f"trial-{trial_index}-bottom{Path(bottom.filename).suffix or '.jpg'}"
                await top.save(top_path)
                await bottom.save(bottom_path)

                screenshot_results = []
                failures = []
                for screenshot_type, image_path in (("top", top_path), ("bottom", bottom_path)):
                    try:
                        screenshot_results.append(
                            await asyncio.to_thread(
                                bot.ocr_service.process_image,
                                store.user_id,
                                image_path,
                                screenshot_type,
                                tmp,
                                candidate_names=candidate_names,
                                manual=manual,
                            )
                        )
                    except OCRFailure as exc:
                        failures.append(f"{screenshot_type}: {exc.message}")
                        failure_details.append(f"Trial {trial_index} {screenshot_type}:\n{exc.to_user_message()}")
                if failures:
                    trial_results.append(None)
                    trial_warnings.append(f"Trial {trial_index} was skipped ({'; '.join(failures)}).")
                    continue

                rows = bot.ocr_service.merge_rows(result.rows for result in screenshot_results)
                issues = ocr_bijection_issues(store, rows)
                if issues:
                    trial_results.append(None)
                    trial_warnings.append(f"Trial {trial_index} was skipped ({'; '.join(issues)}).")
                    continue
                result = add_records_from_ocr(store, rows, interaction.created_at)
                trial_results.append(result)
                trial_warnings.extend(f"Trial {trial_index}: {warning}" for warning in result.warnings)

            for table in format_ocr_trial_tables(team_entries, trial_results):
                await respond(interaction, table)
            for warning_message in format_bullet_messages("Warnings:", trial_warnings):
                await respond(interaction, warning_message)
            for failure_detail in failure_details:
                await respond(interaction, failure_detail)
    except TeamError as exc:
        await respond(interaction, str(exc))


@bot.tree.command(name="ocr", description="Read up to five top and bottom screenshot pairs and append records.")
@app_commands.describe(
    top_image="Trial 1 top screenshot",
    bottom_image="Trial 1 bottom screenshot",
    top_image_2="Trial 2 top screenshot",
    bottom_image_2="Trial 2 bottom screenshot",
    top_image_3="Trial 3 top screenshot",
    bottom_image_3="Trial 3 bottom screenshot",
    top_image_4="Trial 4 top screenshot",
    bottom_image_4="Trial 4 bottom screenshot",
    top_image_5="Trial 5 top screenshot",
    bottom_image_5="Trial 5 bottom screenshot",
    manual="Use your saved /change-ocr regions instead of automatic detection",
)
async def ocr(
    interaction: discord.Interaction,
    top_image: discord.Attachment,
    bottom_image: discord.Attachment,
    top_image_2: Optional[discord.Attachment] = None,
    bottom_image_2: Optional[discord.Attachment] = None,
    top_image_3: Optional[discord.Attachment] = None,
    bottom_image_3: Optional[discord.Attachment] = None,
    top_image_4: Optional[discord.Attachment] = None,
    bottom_image_4: Optional[discord.Attachment] = None,
    top_image_5: Optional[discord.Attachment] = None,
    bottom_image_5: Optional[discord.Attachment] = None,
    manual: bool = False,
) -> None:
    candidate_pairs = [
        (top_image, bottom_image),
        (top_image_2, bottom_image_2),
        (top_image_3, bottom_image_3),
        (top_image_4, bottom_image_4),
        (top_image_5, bottom_image_5),
    ]
    image_pairs: list[tuple[discord.Attachment, discord.Attachment]] = []
    found_gap = False
    for trial_index, (top, bottom) in enumerate(candidate_pairs, start=1):
        if top is None and bottom is None:
            found_gap = True
            continue
        if top is None or bottom is None:
            missing = "top" if top is None else "bottom"
            await respond(interaction, f"Trial {trial_index} is missing its {missing} screenshot. Images must be supplied as complete top + bottom pairs.")
            return
        if found_gap:
            await respond(interaction, "Screenshot pairs must be filled consecutively without skipping a trial.")
            return
        image_pairs.append((top, bottom))
    await _run_ocr_trials(interaction, image_pairs, manual=manual)


@bot.tree.command(name="ocr2", description="Read exactly two top and bottom screenshot pairs and append records.")
@app_commands.describe(
    top_1="Trial 1 top screenshot",
    bottom_1="Trial 1 bottom screenshot",
    top_2="Trial 2 top screenshot",
    bottom_2="Trial 2 bottom screenshot",
    manual="Use your saved /change-ocr regions instead of automatic detection",
)
async def ocr2(
    interaction: discord.Interaction,
    top_1: discord.Attachment,
    bottom_1: discord.Attachment,
    top_2: discord.Attachment,
    bottom_2: discord.Attachment,
    manual: bool = False,
) -> None:
    await _run_ocr_trials(interaction, [(top_1, bottom_1), (top_2, bottom_2)], manual=manual)


@bot.tree.command(name="ocr3", description="Read exactly three top and bottom screenshot pairs and append records.")
@app_commands.describe(
    top_1="Trial 1 top screenshot",
    bottom_1="Trial 1 bottom screenshot",
    top_2="Trial 2 top screenshot",
    bottom_2="Trial 2 bottom screenshot",
    top_3="Trial 3 top screenshot",
    bottom_3="Trial 3 bottom screenshot",
    manual="Use your saved /change-ocr regions instead of automatic detection",
)
async def ocr3(
    interaction: discord.Interaction,
    top_1: discord.Attachment,
    bottom_1: discord.Attachment,
    top_2: discord.Attachment,
    bottom_2: discord.Attachment,
    top_3: discord.Attachment,
    bottom_3: discord.Attachment,
    manual: bool = False,
) -> None:
    await _run_ocr_trials(interaction, [(top_1, bottom_1), (top_2, bottom_2), (top_3, bottom_3)], manual=manual)


@bot.tree.command(name="ocr4", description="Read exactly four top and bottom screenshot pairs and append records.")
@app_commands.describe(
    top_1="Trial 1 top screenshot",
    bottom_1="Trial 1 bottom screenshot",
    top_2="Trial 2 top screenshot",
    bottom_2="Trial 2 bottom screenshot",
    top_3="Trial 3 top screenshot",
    bottom_3="Trial 3 bottom screenshot",
    top_4="Trial 4 top screenshot",
    bottom_4="Trial 4 bottom screenshot",
    manual="Use your saved /change-ocr regions instead of automatic detection",
)
async def ocr4(
    interaction: discord.Interaction,
    top_1: discord.Attachment,
    bottom_1: discord.Attachment,
    top_2: discord.Attachment,
    bottom_2: discord.Attachment,
    top_3: discord.Attachment,
    bottom_3: discord.Attachment,
    top_4: discord.Attachment,
    bottom_4: discord.Attachment,
    manual: bool = False,
) -> None:
    await _run_ocr_trials(
        interaction,
        [(top_1, bottom_1), (top_2, bottom_2), (top_3, bottom_3), (top_4, bottom_4)],
        manual=manual,
    )


@bot.tree.command(name="ocr5", description="Read exactly five top and bottom screenshot pairs and append records.")
@app_commands.describe(
    top_1="Trial 1 top screenshot",
    bottom_1="Trial 1 bottom screenshot",
    top_2="Trial 2 top screenshot",
    bottom_2="Trial 2 bottom screenshot",
    top_3="Trial 3 top screenshot",
    bottom_3="Trial 3 bottom screenshot",
    top_4="Trial 4 top screenshot",
    bottom_4="Trial 4 bottom screenshot",
    top_5="Trial 5 top screenshot",
    bottom_5="Trial 5 bottom screenshot",
    manual="Use your saved /change-ocr regions instead of automatic detection",
)
async def ocr5(
    interaction: discord.Interaction,
    top_1: discord.Attachment,
    bottom_1: discord.Attachment,
    top_2: discord.Attachment,
    bottom_2: discord.Attachment,
    top_3: discord.Attachment,
    bottom_3: discord.Attachment,
    top_4: discord.Attachment,
    bottom_4: discord.Attachment,
    top_5: discord.Attachment,
    bottom_5: discord.Attachment,
    manual: bool = False,
) -> None:
    await _run_ocr_trials(
        interaction,
        [
            (top_1, bottom_1),
            (top_2, bottom_2),
            (top_3, bottom_3),
            (top_4, bottom_4),
            (top_5, bottom_5),
        ],
        manual=manual,
    )


@bot.tree.command(name="stitch", description="Stitch two to ten vertically-scrolled screenshots in order.")
@app_commands.describe(
    image_1="First screenshot (top of the scroll)",
    image_2="Second screenshot",
    image_3="Optional third screenshot",
    image_4="Optional fourth screenshot",
    image_5="Optional fifth screenshot",
    image_6="Optional sixth screenshot",
    image_7="Optional seventh screenshot",
    image_8="Optional eighth screenshot",
    image_9="Optional ninth screenshot",
    image_10="Optional tenth screenshot",
    debug="Include pairwise alignment overlays",
    crop_auto="Crop the final image to the detected scrolling region",
)
async def stitch(
    interaction: discord.Interaction,
    image_1: discord.Attachment,
    image_2: discord.Attachment,
    image_3: Optional[discord.Attachment] = None,
    image_4: Optional[discord.Attachment] = None,
    image_5: Optional[discord.Attachment] = None,
    image_6: Optional[discord.Attachment] = None,
    image_7: Optional[discord.Attachment] = None,
    image_8: Optional[discord.Attachment] = None,
    image_9: Optional[discord.Attachment] = None,
    image_10: Optional[discord.Attachment] = None,
    debug: bool = False,
    crop_auto: bool = False,
) -> None:
    candidates = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10]
    attachments = []
    found_gap = False
    for index, attachment in enumerate(candidates, start=1):
        if attachment is None:
            found_gap = True
            continue
        if found_gap:
            await respond(interaction, f"Images must be filled consecutively; image_{index} was provided after an empty image field.")
            return
        attachments.append(attachment)

    await interaction.response.defer(thinking=True)
    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
        tmp = Path(tmp_name)
        image_paths = []
        try:
            settings = parse_stitch_settings(get_stitch_setting(str(interaction.user.id)))
            for index, attachment in enumerate(attachments, start=1):
                suffix = Path(attachment.filename).suffix or ".img"
                image_path = tmp / f"input-{index:02d}{suffix}"
                await attachment.save(image_path)
                image_paths.append(image_path)

            result = await asyncio.to_thread(
                stitch_image_sequence,
                image_paths,
                crop_top=settings.crop_top,
                crop_bottom=settings.crop_bottom,
                window_height=settings.window_height,
                similarity_threshold=settings.similarity_fraction,
                crop_auto=crop_auto,
                debug=debug,
                debug_dir=tmp / "debug",
            )
            output_path = tmp / "stitched.png"
            await asyncio.to_thread(result.image.save, output_path, "PNG", optimize=True)
            output_paths = [output_path, *result.debug_paths]
            files, prepared = await prepare_generated_image_files(interaction, output_paths)
            content = (
                f"Stitched {len(attachments)} images into {result.image.width} x {result.image.height} pixels."
                + image_compression_note(prepared)
            )
            await respond_files(
                interaction,
                content,
                files=files,
            )
        except StitchingError as exc:
            partial_path = tmp / "partial-stitch.png"
            failed_path = tmp / f"failed-image-{exc.image_index:02d}.png"
            await asyncio.to_thread(exc.partial_stitch.save, partial_path, "PNG", optimize=True)
            await asyncio.to_thread(exc.failed_image.save, failed_path, "PNG", optimize=True)
            diagnostic_paths = [*exc.debug_paths[-8:], partial_path, failed_path]
            files, prepared = await prepare_generated_image_files(interaction, diagnostic_paths)
            message = f"Could not stitch the screenshots: {exc}" + image_compression_note(prepared)
            await respond_files(interaction, message, files=files)
        except (OSError, ValueError) as exc:
            await respond(interaction, f"Could not stitch the screenshots: {exc}")


@bot.tree.command(name="change-stitch", description="Show or update your screenshot stitching settings.")
@app_commands.describe(
    crop_top="Pixels cropped from the top of the final image",
    crop_bottom="Pixels cropped from the bottom of the final image",
    window_height="Comparison window height in pixels",
    similarity_threshold="Required similarity percentage from 0 to 100",
)
async def change_stitch(
    interaction: discord.Interaction,
    crop_top: Optional[int] = None,
    crop_bottom: Optional[int] = None,
    window_height: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
) -> None:
    values = get_stitch_setting(str(interaction.user.id))
    updates = {
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "window_height": window_height,
        "similarity_threshold": similarity_threshold,
    }
    values.update({key: value for key, value in updates.items() if value is not None})
    try:
        settings = parse_stitch_settings(values)
    except ValueError as exc:
        await respond(interaction, str(exc))
        return
    set_stitch_setting(str(interaction.user.id), settings.as_dict())
    await respond(interaction, "Stitch settings updated:\n" + format_stitch_settings(settings))


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

    with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
        tmp = Path(tmp_name)
        image_path = tmp / f"ocr{Path(image.filename).suffix or '.jpg'}"
        await image.save(image_path)
        updating_manual_region = any(value is not None for value in provided)
        coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y) if updating_manual_region else None
        modes = [(True, coords)] if updating_manual_region else [(False, None), (True, None)]
        for manual, mode_coords in modes:
            message, files = await prepare_change_ocr_response(
                interaction,
                str(interaction.user.id),
                image_path,
                screenshot_type,
                tmp,
                manual=manual,
                update_coords=mode_coords,
            )
            await respond_files(interaction, message, files=files)


@bot.tree.command(name="get-records", description="Export score records as a CSV.")
@app_commands.rename(record_filter="filter")
@app_commands.describe(
    user="Discord user, defaults to you",
    record_filter="Uma name/ID, current, or all, plus optional dash-separated specifiers",
)
async def get_records(interaction: discord.Interaction, record_filter: str = "all", user: Optional[discord.User] = None) -> None:
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        store = store_for(target)
        rows = build_records_export(store, record_filter, bot.name_matcher)
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
            build_boxplot(store_for(target), path, sort, order, bot.name_matcher)
            files, prepared = await prepare_generated_image_files(interaction, [path])
            await respond(
                interaction,
                f"Box and whisker for {target_label}" + image_compression_note(prepared),
                file=files[0],
            )
    except TeamError as exc:
        await respond(interaction, f"Box and whisker for {target_label}\n{exc}")


@bot.tree.command(name="box-and-whisker-custom", description="Plot score distributions for a custom set of umas and configurations.")
@app_commands.rename(merge_same_uma="merge-same-uma")
@app_commands.choices(sort=sort_choices, order=order_choices)
@app_commands.describe(
    umas="Comma-separated Uma names/IDs, current, or all; use dashes before specifiers",
    merge_same_uma="Merge each requested selector into one column",
    sort="Sort metric",
    order="ascending or descending",
    date_after="Only include Umas acquired on/after this date (MM/DD/YYYY or MM-DD-YYYY)",
    user="Discord user, defaults to you",
)
async def box_and_whisker_custom(
    interaction: discord.Interaction,
    umas: str,
    merge_same_uma: bool = False,
    sort: str = "median",
    order: str = "descending",
    date_after: Optional[str] = None,
    user: Optional[discord.User] = None,
) -> None:
    await interaction.response.defer(thinking=True)
    target = target_user_or_sender(interaction, user)
    target_label = user_label(target)
    try:
        with tempfile.TemporaryDirectory(dir=config.TMP_DIR) as tmp_name:
            path = Path(tmp_name) / "box-and-whisker-custom.png"
            build_custom_boxplot(
                store_for(target),
                path,
                umas,
                merge_same_uma,
                sort,
                order,
                bot.name_matcher,
                date_after,
            )
            files, prepared = await prepare_generated_image_files(interaction, [path])
            await respond(
                interaction,
                f"Custom box and whisker for {target_label}" + image_compression_note(prepared),
                file=files[0],
            )
    except TeamError as exc:
        await respond(interaction, f"Custom box and whisker for {target_label}\n{exc}")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    if isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, (TeamError, ValueError)):
        await respond(interaction, str(error.original))
        return
    await respond(interaction, f"Something went wrong while running that command: {error}")


def main() -> None:
    thumbnails = download_missing_thumbnails(REFERENCE_ROWS)
    print(f"Uma thumbnails: downloaded {thumbnails.downloaded}, skipped {thumbnails.skipped}, failed {len(thumbnails.failures)}")
    for failure in thumbnails.failures[:10]:
        print(f"Thumbnail download failed: {failure}")
    if len(thumbnails.failures) > 10:
        print(f"... and {len(thumbnails.failures) - 10} more thumbnail download failures")
    token = config.read_discord_token()
    bot.run(token)


if __name__ == "__main__":
    main()
