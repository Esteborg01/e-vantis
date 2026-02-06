# Convención Markdown Clínica — E-VANTIS v1  
**FASE 4 — Capa Editorial (Experimental)**

**Estado:** Propuesta editorial aislada  
**Impacto técnico:** Nulo (no runtime, no imports)  
**Objetivo:** Normalizar la *intencionalidad clínica* del contenido generado sin modificar backend ni UI.

---

## 1. Propósito del documento

Este documento define una **convención editorial explícita** para enriquecer el Markdown clínico de E-VANTIS con *señales semánticas humanas*, sin depender de inferencias automáticas ni cambios técnicos.

La convención:

- ✅ Es **opcional**
- ✅ Es **versionable**
- ✅ Es **auditable**
- ❌ No es obligatoria para producción
- ❌ No forma parte del backend

Su función es permitir responder a la pregunta:

> “¿Este contenido está clínicamente bien priorizado?”

---

## 2. Principios editoriales

1. **La estructura clínica ya existe**  
   Esta convención NO reemplaza las 11 secciones clínicas E-VANTIS.

2. **La semántica debe ser explícita, no inferida**  
   El autor (humano o IA) declara qué es crítico.

3. **La prioridad clínica es editorial, no visual**  
   La UI puede ignorar completamente esta convención.

4. **Todo debe seguir siendo Markdown válido**  
   No se introducen sintaxis incompatibles.

---

## 3. Primitivas editoriales soportadas (v1)

### 3.1 HIGH-YIELD (énfasis clínico)

#### Sintaxis
```md
==Texto high-yield==


