# User Interface Specification
# MidiDrumiGen v2.0 - UI/UX Design Document

**Version:** 2.0.0
**Date:** 2025-11-17
**Status:** Design Phase
**Target Platform:** Max for Live (Ableton Live 11+)

---

## Table of Contents

1. [Overview](#overview)
2. [Max for Live Device](#max-for-live-device)
3. [User Flows](#user-flows)
4. [Component Specifications](#component-specifications)
5. [Visual Design](#visual-design)
6. [Interaction Patterns](#interaction-patterns)
7. [Error States](#error-states)
8. [Future: Web Dashboard](#future-web-dashboard)

---

## Overview

### Design Principles

1. **Simplicity First:** Core action (generate patterns) requires minimal steps
2. **Immediate Feedback:** Progress and status always visible
3. **Non-Disruptive:** Never crashes or freezes Ableton
4. **Informative:** Users always know what's happening and why
5. **Accessible:** Clear labels, readable fonts, WCAG 2.1 AA compliant

### Primary Use Case

1. User types artist name
2. Clicks "Generate"
3. System researches (if needed) and generates patterns
4. MIDI clips appear in Ableton
5. User starts creating music

**Target Time:** < 5 minutes for first-time users

---

## Max for Live Device

### Device Layout (375px × 600px)

```
┌─────────────────────────────────────────────────────────────┐
│  MidiDrumiGen v2.0                               [⚙️] [?]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎤 Artist Name                                            │
│  ┌────────────────────────────────────────────────┐  [🔍] │
│  │ John Bonham                                    │        │
│  └────────────────────────────────────────────────┘        │
│                                                             │
│  💾 Cache Status: ✓ Cached | Last Updated: 2 days ago     │
│  📊 Confidence: ●●●●◐ (0.82) | 24 sources                 │
│                                                             │
│ ┌──────────────────────────────────────────────────────────┤
│ │  Status: Ready to generate                              │
│ │  [████████████████████████████████████] 100%           │
│ └──────────────────────────────────────────────────────────┤
│                                                             │
│  ⚙️ Generation Parameters                                  │
│  ┌────────────┬────────────┬────────────┬────────────┐    │
│  │ Bars       │ Tempo      │ Time Sig   │ Variations │    │
│  │ [ 4  ▼]    │ [120 BPM]  │ [4/4  ▼]   │ [ 4  ▼]    │    │
│  └────────────┴────────────┴────────────┴────────────┘    │
│                                                             │
│  🎛️ Style Controls                                         │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Humanization:  [●────────] 50%                      │  │
│  │ Complexity:    [──────●──] 75%                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────┐  ┌─────────────────────────┐│
│  │     🎲 GENERATE          │  │  🔄 Augment Research   ││
│  │                           │  │  (Add More Sources)    ││
│  └──────────────────────────┘  └─────────────────────────┘│
│                                                             │
│  💡 Tip: Try "Artists like [name]" for similar styles     │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Header Section
- **Device Name:** "MidiDrumiGen v2.0" (left-aligned)
- **Settings Icon (⚙️):** Opens preferences panel
- **Help Icon (?):** Opens documentation

#### 2. Artist Input Section
```
┌─────────────────────────────────────────────┐
│  🎤 Artist Name                            │
│  ┌──────────────────────────────────┐ [🔍] │
│  │ [Input field with autocomplete] │      │
│  └──────────────────────────────────┘      │
└─────────────────────────────────────────────┘
```

**Features:**
- **Text Input:** Single-line text field
- **Autocomplete:** Dropdown suggests cached artists as user types
- **Search Button (🔍):** Triggers research lookup
- **Character Limit:** 100 characters
- **Validation:** Real-time, shows red border if invalid

**Autocomplete Behavior:**
- Appears after 2+ characters typed
- Shows top 5 matches from cached artists
- Can be dismissed with ESC
- Click or Enter to select

#### 3. Status Display Section
```
┌───────────────────────────────────────────────┐
│ 💾 Cache Status: ✓ Cached                   │
│    Last Updated: 2 days ago                  │
│ 📊 Confidence: ●●●●◐ (0.82)                 │
│    Sources: 5 papers, 12 articles, 3 audio  │
└───────────────────────────────────────────────┘
```

**Cache Status States:**
- ✓ Cached (green text)
- ⏳ Researching... (yellow text)
- ✗ Not Found (red text)
- ⚠️ Low Confidence (orange text)

**Confidence Visualization:**
- 0.0-0.2: ●◯◯◯◯ (red)
- 0.2-0.4: ●●◯◯◯ (orange)
- 0.4-0.6: ●●●◯◯ (yellow)
- 0.6-0.8: ●●●●◯ (light green)
- 0.8-1.0: ●●●●● (green)

#### 4. Progress Bar
```
┌───────────────────────────────────────────────┐
│ Status: Researching artist...                │
│ [████████████░░░░░░░░░░░░] 60%              │
│ Current: Analyzing audio (3/5)               │
└───────────────────────────────────────────────┘
```

**States:**
- **Ready:** Hidden or shows "Ready to generate"
- **Research:** Shows collection progress (0-100%)
- **Generation:** Shows "Generating patterns..." (indeterminate spinner)
- **Complete:** Shows "✓ Complete!" (green, 2 sec)
- **Error:** Shows error message (red)

**Progress Steps (Research):**
- 0-25%: Searching papers...
- 25-50%: Scraping articles...
- 50-75%: Analyzing audio...
- 75-100%: Building profile...

#### 5. Generation Parameters
```
┌────────────┬────────────┬────────────┬────────────┐
│ Bars       │ Tempo      │ Time Sig   │ Variations │
│ [ 4  ▼]    │ [120 BPM]  │ [4/4  ▼]   │ [ 4  ▼]    │
└────────────┴────────────┴────────────┴────────────┘
```

**Bars Dropdown:**
- Options: 1, 2, 4, 8, 16
- Default: 4
- Tooltip: "Number of bars to generate"

**Tempo Number Box:**
- Range: 40-300 BPM
- Default: Artist's typical tempo (if cached) or 120
- Tooltip: "Tempo in beats per minute"
- Shows "(Artist typical: 85-95)" when available

**Time Signature Dropdown:**
- Options: 4/4, 3/4, 5/4, 6/8, 7/8
- Default: 4/4
- Tooltip: "Time signature of pattern"

**Variations Dropdown:**
- Options: 1, 2, 4, 6, 8
- Default: 4
- Tooltip: "Number of variations to generate"

#### 6. Style Controls (Advanced)
```
┌───────────────────────────────────────────────┐
│ 🎛️ Style Controls                            │
│                                               │
│ Humanization:  [●────────] 50%               │
│ Complexity:    [──────●──] 75%               │
└───────────────────────────────────────────────┘
```

**Humanization Slider:**
- Range: 0-100%
- Default: 50%
- 0%: Perfectly quantized
- 100%: Maximum timing/velocity variation
- Tooltip: "How 'human' the pattern should feel"

**Complexity Slider:**
- Range: 0-100%
- Default: 75%
- 0%: Simple, repetitive patterns
- 100%: Complex, varied patterns
- Tooltip: "Pattern complexity and variation"

#### 7. Action Buttons

**Generate Button (Primary):**
```
┌──────────────────────────┐
│    🎲 GENERATE           │
│                          │
└──────────────────────────┘
```
- **Size:** Large (200px × 60px)
- **Color:** Blue (#4A90E2)
- **States:**
  - Normal: Blue, clickable
  - Hover: Lighter blue (#5FA3F5)
  - Active: Darker blue (#3A7BC8)
  - Disabled: Gray, shows "Research first" tooltip
  - Loading: Shows spinner, text "Generating..."

**Augment Button (Secondary):**
```
┌──────────────────────────┐
│  🔄 Augment Research     │
│  (Add More Sources)      │
└──────────────────────────┘
```
- **Size:** Medium (200px × 50px)
- **Color:** Gray (#666)
- **Enabled:** Only if artist is cached
- **Action:** Adds 5+ more sources to improve profile

#### 8. Tips/Help Section
```
┌───────────────────────────────────────────────┐
│ 💡 Tip: Try "Artists like [name]" for similar│
│         styles                                │
└───────────────────────────────────────────────┘
```

**Rotating Tips:**
- Tip 1: "Try 'Artists like [name]' for similar styles"
- Tip 2: "Higher confidence = better style accuracy"
- Tip 3: "Augment research to improve low confidence"
- Tip 4: "Use tempo override for creative experimentation"
- Tip 5: "8 variations give more options to choose from"

---

## User Flows

### Flow 1: Generate Pattern for Cached Artist (Happy Path)

```
User Opens Device
       ↓
[Types artist name: "John Bonham"]
       ↓
[Autocomplete shows: ✓ John Bonham (cached)]
       ↓
User Selects from Autocomplete
       ↓
Status Shows: "✓ Cached | Confidence: 0.82"
       ↓
[Optionally adjusts parameters]
       ↓
User Clicks "GENERATE"
       ↓
Progress Bar: "Generating patterns..." [spinner]
       ↓
(2 seconds later)
       ↓
Status: "✓ Complete! 4 clips created"
       ↓
[4 MIDI clips appear in Ableton clip slots]
       ↓
User Can Now Drag/Edit Clips
```

**Total Time:** ~10 seconds

---

### Flow 2: Generate Pattern for New Artist (Research Required)

```
User Opens Device
       ↓
[Types artist name: "Travis Barker"]
       ↓
[Autocomplete shows: "Not cached - will research"]
       ↓
User Clicks "GENERATE"
       ↓
Dialog: "This artist needs research (5-20 min). Continue?"
  [Yes] [No]
       ↓
User Clicks "Yes"
       ↓
Status: "⏳ Researching Travis Barker..."
Progress: [███░░░░░░░░] 25%
"Searching papers..."
       ↓
(5 minutes later)
       ↓
Progress: [█████░░░░░░] 50%
"Scraping articles..."
       ↓
(10 minutes later)
       ↓
Progress: [████████░░░] 75%
"Analyzing audio..."
       ↓
(15 minutes later)
       ↓
Status: "✓ Research complete! Confidence: 0.76"
Progress: [███████████] 100%
       ↓
Status: "Generating patterns..." [spinner]
       ↓
(2 seconds later)
       ↓
Status: "✓ Complete! 4 clips created"
       ↓
[4 MIDI clips appear in Ableton]
```

**Total Time:** 15-20 minutes (research) + 10 seconds (generation)

---

### Flow 3: Augment Existing Artist

```
User Has Cached Artist: "Questlove"
Status: "✓ Cached | Confidence: 0.65 (Low)"
       ↓
User Clicks "Augment Research"
       ↓
Dialog: "Add more sources to improve quality? (~5 min)"
  [Yes] [No]
       ↓
User Clicks "Yes"
       ↓
Status: "⏳ Augmenting Questlove..."
Progress: [████░░░░░░░] 35%
"Collecting additional sources..."
       ↓
(5 minutes later)
       ↓
Status: "✓ Augmentation complete!"
"Confidence improved: 0.65 → 0.81"
"Added: 3 papers, 5 articles, 2 audio"
       ↓
User Can Now Generate with Higher Confidence
```

---

## Component Specifications

### Typography

**Font Family:** Ableton Sans (or fallback: Arial, Helvetica, sans-serif)

**Font Sizes:**
- **Title:** 18px, bold (MidiDrumiGen v2.0)
- **Section Headers:** 14px, semi-bold (🎤 Artist Name, ⚙️ Parameters)
- **Body Text:** 12px, regular (status messages, labels)
- **Small Text:** 10px, regular (tooltips, tips)
- **Input Text:** 13px, regular

**Line Height:** 1.4 for body text, 1.2 for headers

---

### Color Palette

**Primary Colors:**
```
Background:   #2B2B2B (Dark Gray)
Surface:      #3A3A3A (Medium Gray)
Border:       #4A4A4A (Light Gray)
Text:         #CCCCCC (Light Gray)
Text Dim:     #999999 (Dim Gray)
```

**Accent Colors:**
```
Primary:      #4A90E2 (Blue) - Generate button, progress
Success:      #7ED321 (Green) - Cached, complete
Warning:      #F5A623 (Orange) - Low confidence
Error:        #D0021B (Red) - Errors, failures
Info:         #50E3C2 (Teal) - Tips, help
```

**State Colors:**
```
Hover:        #5FA3F5 (Light Blue)
Active:       #3A7BC8 (Dark Blue)
Disabled:     #555555 (Dark Gray)
Focus:        #4A90E2 (Blue) - 2px outline
```

---

### Icons

**Icon Library:** Material Icons or Font Awesome

**Icon Sizes:**
- Small: 14px (inline with text)
- Medium: 18px (buttons)
- Large: 24px (primary actions)

**Used Icons:**
- 🎤 Artist (fa-microphone)
- 💾 Cache (fa-database)
- 📊 Confidence (fa-chart-bar)
- ⚙️ Settings (fa-cog)
- ? Help (fa-question-circle)
- 🔍 Search (fa-search)
- 🎲 Generate (fa-dice)
- 🔄 Augment (fa-sync)
- ✓ Success (fa-check)
- ✗ Error (fa-times)
- ⏳ Loading (fa-spinner, animated)

---

### Spacing

**Padding:**
- Container: 16px
- Section: 12px
- Component: 8px
- Inline: 4px

**Margins:**
- Between Sections: 20px
- Between Components: 12px
- Between Elements: 8px

---

### Animations

**Progress Bar:**
- Filled portion animates smoothly (ease-in-out, 0.3s)
- Indeterminate: sliding gradient animation

**Buttons:**
- Hover: scale(1.02), transition: 0.2s
- Click: scale(0.98), transition: 0.1s

**Notifications:**
- Fade in: opacity 0 → 1, 0.3s
- Fade out: opacity 1 → 0, 0.5s (after 3s)

**Loading Spinner:**
- Rotation: 360deg, 1s, infinite

---

## Interaction Patterns

### Artist Input Autocomplete

**Behavior:**
- Triggers after 2 characters typed
- Debounced (300ms delay)
- Shows top 5 matches
- Sorted by: exact match > starts with > contains
- Keyboard navigation: ↑/↓ arrows, Enter to select, Esc to dismiss

**Visual:**
```
┌────────────────────────────────┐
│ John                          │ ← Input field
└────────────────────────────────┘
┌────────────────────────────────┐
│ ✓ John Bonham (0.89)          │ ← Exact match, cached
│ ✓ John Dolmayan (0.72)        │ ← Cached
│   Johnny Rabb                  │ ← Not cached
│   John "JR" Robinson           │
└────────────────────────────────┘
```

### Parameter Tooltips

**Trigger:** Hover over parameter label or value (1s delay)

**Visual:**
```
      Bars
     [ 4 ▼]
       ↓
┌──────────────────────────────────┐
│ Number of bars to generate       │
│ Range: 1-16                      │
│ Default: 4                       │
└──────────────────────────────────┘
```

### Confidence Rating Interaction

**Visual Feedback:**
- **Hover over dots:** Shows numeric score in tooltip
- **Click:** Opens detail panel with source breakdown

**Detail Panel:**
```
┌────────────────────────────────────┐
│ Confidence Score: 0.82             │
├────────────────────────────────────┤
│ Sources:                           │
│  📄 Papers: 5                      │
│  📰 Articles: 12                   │
│  🎵 Audio: 3                       │
│  🎹 MIDI: 4                        │
│                                    │
│ Quality: High (24 sources)         │
│ Last Updated: 2025-11-15           │
│                                    │
│ [View Details] [Augment]           │
└────────────────────────────────────┘
```

---

## Error States

### Error 1: Artist Not Found
```
┌────────────────────────────────────┐
│ ✗ Artist "Unknown Name" not found │
│                                    │
│ Suggestions:                       │
│  • Check spelling                  │
│  • Try full name (e.g., "John     │
│    Bonham" not "Bonham")           │
│  • Use drummer's name, not band    │
│                                    │
│ Similar artists:                   │
│  • John Bonham                     │
│  • Keith Moon                      │
│                                    │
│ [Try Again]                        │
└────────────────────────────────────┘
```

### Error 2: Low Confidence Warning
```
┌────────────────────────────────────┐
│ ⚠️ Low confidence (0.45)           │
│                                    │
│ Limited data found for this artist.│
│ Generation may not match style     │
│ accurately.                        │
│                                    │
│ Options:                           │
│  • [Augment Research] (recommended)│
│  • [Generate Anyway]               │
│  • [Choose Different Artist]       │
└────────────────────────────────────┘
```

### Error 3: Generation Failed
```
┌────────────────────────────────────┐
│ ✗ Generation failed                │
│                                    │
│ All LLM providers are unavailable. │
│                                    │
│ Details:                           │
│  • OpenAI: Rate limit exceeded     │
│  • Anthropic: API error            │
│  • Google: Connection timeout      │
│                                    │
│ [Retry] [Report Issue]             │
└────────────────────────────────────┘
```

### Error 4: Network Error
```
┌────────────────────────────────────┐
│ ✗ Connection error                 │
│                                    │
│ Cannot reach MidiDrumiGen server.  │
│                                    │
│ Check:                             │
│  • API server is running           │
│    (localhost:8000)                │
│  • Network connection              │
│  • Firewall settings               │
│                                    │
│ [Retry] [View Logs]                │
└────────────────────────────────────┘
```

---

## Loading States

### State 1: Researching Artist
```
┌────────────────────────────────────┐
│ ⏳ Researching Travis Barker...    │
│                                    │
│ [████████░░░░░░░░░░░░] 40%        │
│                                    │
│ Current: Scraping articles (8/12)  │
│ Estimated time: 10 minutes         │
│                                    │
│ [Cancel]                           │
└────────────────────────────────────┘
```

### State 2: Generating Patterns
```
┌────────────────────────────────────┐
│ 🎲 Generating patterns...          │
│                                    │
│ [  Spinner animation  ]            │
│                                    │
│ This usually takes 10-30 seconds   │
└────────────────────────────────────┘
```

### State 3: Importing to Ableton
```
┌────────────────────────────────────┐
│ 📥 Importing clips to Ableton...   │
│                                    │
│ [  Spinner animation  ]            │
│                                    │
│ Creating 4 clips in track 1...     │
└────────────────────────────────────┘
```

---

## Success States

### Success 1: Generation Complete
```
┌────────────────────────────────────┐
│ ✓ Success! 4 variations created    │
│                                    │
│ 📁 Clips added to track 1          │
│ ⏱️  Generated in 12 seconds        │
│ 🤖 Provider: OpenAI GPT-4          │
│                                    │
│ [Generate More] [Done]             │
└────────────────────────────────────┘
```

### Success 2: Research Complete
```
┌────────────────────────────────────┐
│ ✓ Research complete!               │
│                                    │
│ Artist: Travis Barker              │
│ Confidence: ●●●●◯ (0.76)           │
│ Sources: 18 total                  │
│                                    │
│ Ready to generate patterns.        │
│                                    │
│ [Generate Now]                     │
└────────────────────────────────────┘
```

---

## Future: Web Dashboard

### Dashboard Overview

**URL:** `http://localhost:8000/dashboard`

**Features:**
1. **Artist Library**
   - Browse all cached artists
   - Sort by: name, confidence, date added
   - Filter by: confidence level, source count
   - Search with autocomplete

2. **Artist Detail Page**
   - View all research sources
   - See quantitative parameters
   - Preview MIDI templates
   - Edit/augment research
   - View generation history

3. **Statistics**
   - Total artists cached
   - Total generations
   - Average confidence score
   - Provider usage breakdown
   - API costs (per provider)

4. **Settings**
   - API key management
   - Default generation parameters
   - Provider preferences (primary/fallback order)
   - Cache management (clear old entries)

5. **Documentation**
   - User guide
   - API reference
   - Troubleshooting
   - FAQ

---

## Accessibility (WCAG 2.1 AA)

### Color Contrast
- **Text on Background:** Minimum 4.5:1 ratio
- **Large Text:** Minimum 3:1 ratio
- **Interactive Elements:** Minimum 3:1 ratio

### Keyboard Navigation
- **Tab Order:** Logical top-to-bottom
- **Focus Indicators:** Visible 2px blue outline
- **Shortcuts:**
  - `Cmd/Ctrl + G`: Generate
  - `Cmd/Ctrl + R`: Augment Research
  - `Cmd/Ctrl + /`: Toggle Help
  - `Esc`: Cancel operation

### Screen Reader Support
- All form inputs have labels
- Status updates are announced
- Progress changes are announced
- Errors are announced immediately

### Motion
- **Reduced Motion:** Respect `prefers-reduced-motion`
- **Animations:** Can be disabled in settings

---

## Responsive Design (Future: Web)

### Breakpoints
- **Mobile:** 320-767px
- **Tablet:** 768-1023px
- **Desktop:** 1024px+

### Mobile Adaptations
- Stack parameters vertically
- Larger touch targets (44px minimum)
- Collapsible sections
- Bottom sheet for actions

---

## Implementation Notes

### Max for Live Specifics

**File:** `ableton/MidiDrumGen.amxd`

**Max Objects Used:**
- `textedit`: Artist input
- `live.button`: Action buttons
- `live.slider`: Progress bar, parameter sliders
- `live.numbox`: Numeric inputs (tempo, bars)
- `live.menu`: Dropdowns (time signature, variations)
- `live.text`: Status display, labels
- `js`: JavaScript bridge for API calls
- `live.path`: MIDI clip import

**JavaScript Bridge:**
```javascript
// ableton/js/bridge.js
const API_URL = "http://localhost:8000/api/v1";

// Called when user clicks Generate
async function generatePattern(artistName, params) {
    updateStatus("Checking cache...");

    const cached = await checkCache(artistName);

    if (!cached) {
        const confirm = await showDialog(
            "Research Required",
            "This artist needs research (5-20 min). Continue?",
            ["Yes", "No"]
        );

        if (confirm !== "Yes") return;

        await researchArtist(artistName);
    }

    updateStatus("Generating patterns...");
    const result = await generate(artistName, params);

    updateStatus("Importing clips...");
    await importClipsToLive(result.midi_files);

    updateStatus("✓ Complete! " + result.midi_files.length + " clips created");
}
```

---

## Design Assets

### Required Assets
- MidiDrumiGen logo (SVG, 32×32px)
- Icon set (Material Icons, 14-24px)
- Loading spinner animation (GIF or CSS)
- Sample screenshots for documentation

### Deliverables
- Max for Live device (.amxd)
- JavaScript bridge code (.js)
- CSS stylesheets (for web dashboard)
- Design mockups (Figma/Sketch)
- User testing reports

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Next Review:** 2025-12-01
**Contact:** UI/UX Team
