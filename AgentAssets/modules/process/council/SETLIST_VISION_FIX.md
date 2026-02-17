# Setlist Vision Fix: Council Decision

**Date:** December 2025  
**Issue:** Setlists are still being treated as separate entities that need to be "loaded" rather than being an integral part of projects.

---

## Problem Statement

The current UI still has an "Existing Setlists" dropdown that treats setlists as separate entities. This violates the core vision that **setlists are built into projects, not separate entities**.

**Current (Wrong) Mental Model:**
- User opens project
- User must "select" or "load" a setlist from dropdown
- Setlists feel like separate things

**Correct Mental Model:**
- User opens project
- If project has setlists, they appear automatically
- Setlists are part of the project, like blocks or connections
- No "loading" or "selecting" needed

---

## Council Analysis

### Architect Analysis

**Problem Understanding:**
The UI still treats setlists as separate entities that need to be loaded, when they should be automatically displayed as part of the project.

**Key Concerns:**
1. **Separation of concerns violated**: UI suggests setlists are separate from projects
2. **Mental model confusion**: "Load setlist" implies setlists exist independently
3. **Unnecessary abstraction**: Dropdown adds complexity without value

**Vote: Reject Current UI Pattern**

**Reasoning:**
The current UI pattern contradicts the architectural decision that setlists belong to projects. We need to remove the "load setlist" concept entirely.

**Required Changes:**
1. Remove "Existing Setlists" dropdown section
2. Auto-display setlists when project is loaded
3. If project has multiple setlists, show them as tabs or a simple list
4. Make it clear setlists are part of the project

---

### Systems Analysis

**Problem Understanding:**
The current implementation is technically correct (setlists are scoped to projects in the database), but the UI suggests otherwise.

**Key Concerns:**
1. **No technical issues**: Database schema is correct
2. **UI/UX mismatch**: UI doesn't reflect the data model
3. **User confusion**: Users might think setlists are global

**Vote: Approve Fix**

**Reasoning:**
The backend is correct. We just need to fix the UI to match the data model.

---

### UX Analysis

**Problem Understanding:**
Users are confused because the UI suggests setlists need to be "loaded" when they should just appear as part of the project.

**Key Concerns:**
1. **Cognitive load**: Extra step (selecting setlist) adds friction
2. **Mental model mismatch**: UI doesn't match user's understanding
3. **Discoverability**: Setlists should be obvious, not hidden in dropdown

**Vote: Reject Current UI Pattern**

**Reasoning:**
The UX violates the principle that setlists are part of projects. Users shouldn't need to "load" them.

**Required Changes:**
1. When project is loaded, automatically show its setlists
2. If project has one setlist, show it directly
3. If project has multiple setlists, show them as tabs or a list
4. Remove all "load setlist" UI elements

---

### Pragmatic Analysis

**Problem Understanding:**
The current implementation works but has unnecessary UI complexity that confuses users.

**Key Concerns:**
1. **Unnecessary code**: Dropdown and "load" logic adds complexity
2. **Maintenance burden**: Extra UI state to manage
3. **User frustration**: Users keep asking why setlists need to be loaded

**Vote: Approve Simplification**

**Reasoning:**
Removing the "load setlist" concept simplifies the code and matches user expectations.

---

## Unanimous Recommendation

**RECOMMENDATION: Remove "Load Setlist" Concept Entirely**

### Required Changes

1. **Remove "Existing Setlists" Section**
   - Delete `_create_existing_setlists_section()` method
   - Remove dropdown and refresh button
   - Remove `_on_existing_setlist_selected()` handler

2. **Auto-Display Setlists**
   - When project is loaded, automatically load its setlists
   - If project has one setlist, show it directly
   - If project has multiple setlists, show them as tabs or a simple list

3. **Simplify UI Flow**
   - No "select setlist" step
   - Setlists appear automatically when project is loaded
   - Create new setlist → automatically becomes active

4. **Update Mental Model**
   - Setlists are part of projects (like blocks)
   - No "loading" or "selecting" needed
   - They just exist as part of the project

### Implementation Plan

1. **Remove dropdown section** from `setlist_view.py`
2. **Auto-load setlists** when project changes
3. **Show setlists directly** (no selection step)
4. **Handle multiple setlists** with tabs or list view
5. **Update documentation** to reflect new mental model

### Success Criteria

- ✅ No "load setlist" dropdown
- ✅ Setlists appear automatically when project is loaded
- ✅ Clear that setlists are part of the project
- ✅ Simple, intuitive flow

---

## Conclusion

The current UI violates the core vision that setlists are built into projects. We must remove the "load setlist" concept entirely and make setlists appear automatically as part of the project.

**Status: Approved - Immediate Fix Required**

