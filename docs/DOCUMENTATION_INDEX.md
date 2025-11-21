# MidiDrumiGen v2.0 - Documentation Index

**Version:** 2.0.0
**Date:** 2025-11-17
**Status:** Complete

---

## 📚 Core Documentation

### 1. Product Requirements Document (PRD)
**File:** `docs/PRD.md`
**Purpose:** Complete product specification, user stories, success criteria
**Audience:** Product managers, stakeholders, developers

**Contents:**
- Executive Summary
- Problem Statement & Vision
- User Stories (13 detailed stories across 4 epics)
- Functional Requirements (Research, Generation, Database, API)
- Non-Functional Requirements (Performance, Reliability, Security)
- Success Metrics & KPIs
- Risk Assessment
- Release Plan

---

### 2. Architecture Document
**File:** `docs/ARCHITECTURE.md`
**Purpose:** Technical system architecture and implementation details
**Audience:** Developers, system architects, DevOps

**Contents:**
- High-Level Architecture Diagram
- System Components (9 agents)
- Data Flow Diagrams
- Technology Stack (Python 3.11, FastAPI, PostgreSQL, Redis)
- Database Schema (PostgreSQL + pgvector)
- API Specification (REST endpoints)
- Deployment Architecture (Dev & Production)
- Security & Performance Optimization
- Monitoring & Observability

---

### 3. User Interface Specification
**File:** `docs/UI.md`
**Purpose:** UI/UX design for Max for Live device and future web dashboard
**Audience:** UI designers, frontend developers, Max for Live developers

**Contents:**
- Max for Live Device Layout (375×600px)
- Component Specifications (input, progress bar, buttons)
- User Flows (3 primary flows)
- Visual Design (typography, colors, icons, spacing)
- Interaction Patterns (autocomplete, tooltips, feedback)
- Error States & Loading States
- Accessibility (WCAG 2.1 AA)
- Implementation Notes (Max objects, JavaScript bridge)

---

### 4. Orchestrator Meta Prompt
**File:** `docs/ORCHESTRATOR_META_PROMPT.md`
**Purpose:** Comprehensive prompt for AI-assisted development (Claude Code CLI)
**Audience:** AI development agents, automated implementation tools

**Contents:**
- Complete system architecture overview
- 9 sub-agent specifications
- Inter-agent communication protocols
- 6-phase implementation plan
- Technology stack with current versions
- Configuration templates
- Success criteria
- Implementation instructions

---

## 🤖 Sub-Agent Documentation

### 5. Research Orchestrator Agent
**File:** `docs/agents/RESEARCH_ORCHESTRATOR_AGENT.md`
**Purpose:** Coordinates all research collectors to build StyleProfile
**Audience:** Backend developers implementing research pipeline

**Contents:**
- Agent responsibilities
- Class structure and methods
- 4 Collector sub-agents (Papers, Articles, Audio, MIDI)
- Configuration options
- Error handling
- Progress reporting
- Testing strategies
- Augmentation support
- Performance metrics

---

### 6. LLM Provider Manager Agent
**File:** `docs/agents/LLM_PROVIDER_MANAGER_AGENT.md`
**Purpose:** Manages multiple LLM providers with automatic failover
**Audience:** Backend developers implementing generation pipeline

**Contents:**
- Provider interface (abstract base class)
- Manager implementation (failover logic)
- Provider implementations:
  - OpenAI (GPT-4-turbo)
  - Anthropic (Claude 3 Opus/Sonnet)
  - Google (Gemini 1.5 Pro)
- Configuration
- Error handling & retry logic
- Cost tracking
- Performance monitoring
- Testing strategies

---

## 📋 Additional Sub-Agent Specs (To Be Created)

### 7. Style Profile Builder Agent
**File:** `docs/agents/STYLE_PROFILE_BUILDER_AGENT.md` (TODO)
**Purpose:** Aggregates research data into generation-ready StyleProfile

### 8. Template Generator Agent
**File:** `docs/agents/TEMPLATE_GENERATOR_AGENT.md` (TODO)
**Purpose:** Rule-based MIDI generation using templates (fallback)

### 9. Hybrid Coordinator Agent
**File:** `docs/agents/HYBRID_COORDINATOR_AGENT.md` (TODO)
**Purpose:** Combines LLM + template approaches with quality validation

### 10. MIDI Export Agent
**File:** `docs/agents/MIDI_EXPORT_AGENT.md` (TODO)
**Purpose:** Converts JSON to MIDI files with humanization

---

## 🗂️ Legacy Documentation (Phase 6 Weeks 1-2)

### Context Files
Located in `docs/`:

- `DEVELOPMENT_START_GUIDE.md` - Original Week 3 training guide (now superseded)
- `PHASE_6_WEEK3_HANDOFF.md` - Training pipeline spec (now obsolete)
- `PHASE_6_WEEK2_SUMMARY.md` - Style transfer implementation
- `PHASE_6_WEEK1_SUMMARY.md` - Producer research implementation
- `PHASE_6_HANDOFF.md` - Original Phase 6 plan
- `WHAT_WAS_DELIVERED.md` - Week 1-2 deliverables

**Note:** These documents describe the OLD architecture (pre-trained models). They are kept for reference but should not be used for v2.0 implementation.

---

## 🎯 Quick Navigation

### For Product Managers
1. Start with `PRD.md` for complete product vision
2. Review `UI.md` for user experience design
3. Check success metrics in `PRD.md` Section 8

### For Developers
1. Read `ARCHITECTURE.md` for system overview
2. Review specific agent docs in `docs/agents/`
3. Use `ORCHESTRATOR_META_PROMPT.md` for AI-assisted development
4. Check `ARCHITECTURE.md` Section 4 for technology stack

### For UI/UX Designers
1. Start with `UI.md` for complete UI specification
2. Review user flows and error states
3. Check accessibility requirements

### For DevOps/Infrastructure
1. Read `ARCHITECTURE.md` Section 7 (Deployment)
2. Review database schema (Section 5)
3. Check monitoring requirements (Section 10)

### For AI Development Tools (Claude Code CLI, etc.)
1. Load `ORCHESTRATOR_META_PROMPT.md` as primary context
2. Reference `ARCHITECTURE.md` for technical details
3. Use agent docs for specific component implementation

---

## 📊 Documentation Statistics

### Total Documentation
- **Core Docs:** 4 files (~30,000 words)
- **Agent Docs:** 2 files (~8,000 words, 8 more planned)
- **Legacy Docs:** 10+ files (reference only)

### Coverage
- ✅ Product Requirements: 100% (PRD complete)
- ✅ System Architecture: 100% (full architecture documented)
- ✅ UI/UX Design: 100% (Max for Live device spec complete)
- ⏳ Sub-Agent Specs: 20% (2 of 10 agents documented)
- ✅ AI Meta Prompt: 100% (complete implementation guide)

---

## 🚀 Implementation Roadmap

### Phase 1: Documentation (COMPLETE ✅)
- ✅ PRD created
- ✅ Architecture designed
- ✅ UI specified
- ✅ Meta prompt written
- ✅ Key agent specs created

### Phase 2: Infrastructure (Week 1)
- [ ] Setup PostgreSQL + pgvector
- [ ] Create Alembic migrations
- [ ] Archive old training code
- [ ] Setup environment variables
- [ ] Create configuration files

### Phase 3: Research Pipeline (Week 2-3)
- [ ] Implement 4 collectors
- [ ] Build research orchestrator
- [ ] Implement style profile builder
- [ ] Test with 10 artists

### Phase 4: Generation Engine (Week 3-4)
- [ ] Implement 3 LLM providers
- [ ] Build prompt engineering
- [ ] Create template generator
- [ ] Build hybrid coordinator

### Phase 5: MIDI Export (Week 4)
- [ ] JSON to MIDI conversion
- [ ] Humanization pipeline
- [ ] Validation checks

### Phase 6: Ableton Integration (Week 5)
- [ ] Build Max for Live device
- [ ] Implement JavaScript bridge
- [ ] Test clip import

### Phase 7: Testing (Week 6)
- [ ] End-to-end tests
- [ ] Performance validation
- [ ] User acceptance testing

---

## 📝 Documentation Standards

### Versioning
All documents follow semantic versioning:
- **Major:** Breaking changes to architecture
- **Minor:** New features or agents
- **Patch:** Bug fixes, clarifications

### Review Cycle
- **Weekly:** Review accuracy of implementation docs
- **Monthly:** Update architecture for system changes
- **Quarterly:** Review and update PRD

### Change Log
All major documents maintain a change log at the bottom.

---

## 🔗 External References

### Technology Documentation
- **FastAPI:** https://fastapi.tiangolo.com/
- **PostgreSQL:** https://www.postgresql.org/docs/
- **pgvector:** https://github.com/pgvector/pgvector
- **OpenAI API:** https://platform.openai.com/docs/
- **Anthropic API:** https://docs.anthropic.com/
- **Google Gemini:** https://ai.google.dev/docs
- **Max for Live:** https://docs.cycling74.com/max8/vignettes/live_object_model

### Research Sources
- **Semantic Scholar:** https://api.semanticscholar.org/
- **ArXiv:** https://arxiv.org/help/api/
- **MIDI Spec:** https://www.midi.org/specifications

---

## ❓ FAQ

**Q: Where should I start if I'm new to the project?**
A: Read `PRD.md` for product vision, then `ARCHITECTURE.md` for technical overview.

**Q: How do I implement a specific agent?**
A: Check `docs/agents/[AGENT_NAME]_AGENT.md` for detailed spec. If not available, reference `ARCHITECTURE.md` and `ORCHESTRATOR_META_PROMPT.md`.

**Q: What happened to the training pipeline from Week 3?**
A: The training approach (pre-training PyTorch models) was replaced with on-demand research + LLM generation. See `ORCHESTRATOR_META_PROMPT.md` for new architecture.

**Q: Where are the API keys stored?**
A: All API keys are stored in `.env` file (not committed to Git). See `ARCHITECTURE.md` Section 8 (Security).

**Q: Can I use this with DAWs other than Ableton?**
A: Phase 1 focuses on Max for Live (Ableton). VST3 plugin is planned for v3.0 (Month 12).

---

## 📞 Contact & Support

**Project Lead:** [To be assigned]
**Architecture Team:** [To be assigned]
**Documentation Maintainer:** AI Assistant
**Last Updated:** 2025-11-17

---

**Next Steps:**
1. Review all documentation for completeness
2. Begin Phase 2 implementation (Infrastructure setup)
3. Create remaining sub-agent specification documents
4. Setup project repository and CI/CD

**Status:** ✅ Ready for Implementation
