// Okay, let's refactor and enhance the FlowMind application.
//
// Here's a revised version incorporating the requested changes:
//
// *   **Latest LangChain:** Updated `ChatOllama`, introduced `OllamaEmbeddings`, and used the modern `MemoryVectorStore` interface.
// *   **Refactoring:**
// *   Organized code using `namespace` for better logical grouping.
// *   Added JSDoc comments for clarity.
//                              *   Improved type definitions (e.g., `Status` enum).
// *   Refined Datascript schema and queries.
// *   Made UI updates reactive (triggered by DB changes) instead of polling.
// *   Slightly improved UI layout and added Load/Save functionality.
// *   **New Features:**
// *   **Persistence:** Added basic LocalStorage saving/loading for the database.
// *   **Reactive UI:** UI updates efficiently when the database changes.
// *   **Embedding Integration:** Thoughts can be automatically embedded and added to the vector store via rules.
// *   **Memory Search Integration:** Rules can trigger vector store searches.
// *   **Improved User Prompt Handling:** User prompts are handled via dedicated thought statuses and metadata, decoupling the UI tool.
// *   **Rule Synthesis:** Added basic LLM call to synthesize rules (requires manual parsing/integration currently, marked with TODO).
// *   **Debug View:** Added display of Rules in the debug panel.
// *   **Functionality Retained:** Core note-taking, agentic processing via rules, tool execution (LLM, Memory, User Interaction), goal-based scheduling, undo, configuration.
//
//     ```typescript
// FlowMind: Agentic note-taking and planning system, simple yet powerful
// Single-file TypeScript app with intuitive UI, leveraging FOSS dependencies
// Dependencies (npm): react react-dom socket.io-client datascript langchain @langchain/community @langchain/core uuid @types/react @types/react-dom @types/uuid
// Run in browser with CDNs or Node.js with:
// npm install react react-dom socket.io-client datascript langchain @langchain/community @langchain/core uuid
// npm install --save-dev @types/react @types/react-dom @types/uuid typescript ts-loader webpack webpack-cli html-webpack-plugin
// (Then configure webpack or use a simpler setup like parcel/vite)

import * as React from 'react';
import { useState, useEffect, useCallback, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
// import { io, Socket } from 'socket.io-client'; // Collaboration tool commented out for now
import * as ds from 'datascript';
import { v4 as uuid } from 'uuid';
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "@langchain/community/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { BaseLanguageModel } from "@langchain/core/language_models/base";
import { Embeddings } from "@langchain/core/embeddings";
import { VectorStore } from "@langchain/core/vectorstores";


// ======== Global Constants ========
const DB_STORAGE_KEY = 'flowmind_db';
const META_KEYS = {
    UI_PROMPT_TEXT: ':flowmind.meta.key/ui_prompt_text',
    WAITING_FOR_USER_INPUT: ':flowmind.meta.key/waiting_for_user_input',
    PARENT_UUID: ':flowmind.meta.key/parent_uuid',
    SOURCE_RULE_UUID: ':flowmind.meta.key/source_rule_uuid',
    TOOL_RESULT_FOR: ':flowmind.meta.key/tool_result_for', // UUID of thought waiting for this result
    RESPONSE_TO_PROMPT: ':flowmind.meta.key/responded_to_prompt', // UUID of prompt thought
    EMBEDDING_VECTOR: ':flowmind.meta.key/embedding_vector' // Storing embedding ref
};

// ======== Types ========
namespace Types {
    export type UUID = string;

    /** Represents different kinds of terms in the system's logic. */
    export type TermKind = 'atom' | 'var' | 'struct' | 'list';
    /** Represents a logical term, the basic unit of content and rules. */
    export type Term = {
        kind: TermKind;
        name?: string; // For atom, struct, var
        args?: Term[]; // For struct
        elements?: Term[]; // For list
    };

    /** Represents the truth value (confidence) of a thought or rule. */
    export type Truth = { pos: number; neg: number };
    /** Represents the goal associated with a thought (priority, source). */
    export type Goal = { value: number; source: string; time: Date };

    /** Represents the processing status of a thought. */
    export enum Status {
        PENDING = ':flowmind.status/pending',     // Ready for processing
        ACTIVE = ':flowmind.status/active',       // Currently being processed
        WAITING = ':flowmind.status/waiting',     // Waiting for external input (e.g., user, tool)
        DONE = ':flowmind.status/done',           // Processing completed successfully
        FAILED = ':flowmind.status/failed',       // Processing failed
    }
    // Type alias for Status enum values
    export type StatusValue = `${Status}`;


    /** Represents a unit of thought or task in the system. */
    export type Thought = {
        ':db/id'?: number;
        uuid: UUID;
        content: Term; // Reference to Term entity in DB during storage
        contentRef?: number; // DB ref ID for content
        truth: Truth;
        goal: Goal;
        status: StatusValue;
        meta: Map<string, any>; // Can store refs or simple values. Refs need careful handling.
        createdAt: Date;
        modifiedAt: Date;
    };

    /** Represents an inference rule (Head :- Body). */
    export type Rule = {
        ':db/id'?: number;
        uuid: UUID;
        head: Term; // Reference to Term entity
        headRef?: number;
        body: Term[]; // References to Term entities
        bodyRefs?: number[];
        truth: Truth;
        meta: Map<string, any>;
        createdAt: Date;
        modifiedAt: Date;
    };

    /** Represents an action to be performed by the ActionHandler. */
    export type Action = {
        type: 'tool' | 'memory' | 'log';
        name?: string; // Tool name or memory operation
        params?: Term; // Parameters for the action
        context: {
            triggerUUID: UUID; // UUID of the thought that triggered this action
            waitingThoughtUUID?: UUID; // UUID of the thought waiting for a tool result
            resultBindingVar?: string; // Variable name in the rule to bind the result to
            ruleUUID?: UUID; // UUID of the rule that generated this action
        };
    };

    /** Represents a system event. */
    export type Event = {
        ':db/id'?: number;
        uuid: UUID;
        targetUUID?: UUID; // UUID of the entity the event relates to (e.g., Thought, Rule)
        type: string; // e.g., ':flowmind.event.type/tool_success'
        data: any; // Additional event data (can be simple value or ref)
        time: Date;
    };

    /** Interface for executable tools. */
    export interface Tool {
        execute(params: Term): Promise<Term>;
    }

    /** Type for the main configuration object. */
    export interface Config {
        llmEndpoint: string;
        ollamaModel: string;
        vectorStore: 'memory'; // Extendable in the future
        syncInterval: number; // UI refresh interval (fallback if direct reactivity fails)
        autoSaveInterval: number; // Interval for auto-saving DB to localStorage
    }

    /** Type for pending user prompts displayed in the UI. */
    export type UIPrompt = {
        promptThoughtUUID: UUID;
        text: string;
        waitingThoughtUUID: UUID; // The thought that generated this prompt and is waiting
    };
}

// ======== Configuration ========
const DEFAULT_CONFIG: Types.Config = {
    llmEndpoint: 'http://localhost:11434',
    ollamaModel: 'llama3:latest', // Or whichever model you have loaded in Ollama
    vectorStore: 'memory',
    syncInterval: 2000, // Less critical now with reactivity, used as fallback/refresh
    autoSaveInterval: 30000, // Save every 30 seconds
};

// ======== Term Utilities ========
namespace TermUtils {
    import Term = Types.Term;
    import DBInterface = Database.DBInterface;

    /** Converts a Datascript entity map into a Term object. Requires pulling referenced terms. */
    export const entityToTerm = (db: DBInterface, entityId: number): Term | null => {
        if (!entityId) return null;
        const entity = db.pull(entityId, '[*]');
        if (!entity) return null;

        const kind = entity[':term/kind']?.replace(':flowmind.term/', '') as Types.TermKind;
        const name = entity[':term/name'];

        switch (kind) {
            case 'struct':
                const argIds = entity[':term/args'] || [];
                return {
                    kind: 'struct',
                    name: name,
                    args: argIds.map((ref: any) => entityToTerm(db, ref[':db/id'])).filter(Boolean) as Term[],
                };
            case 'list':
                const elementIds = entity[':term/elements'] || [];
                return {
                    kind: 'list',
                    elements: elementIds.map((ref: any) => entityToTerm(db, ref[':db/id'])).filter(Boolean) as Term[],
                };
            case 'atom':
            case 'var':
                return { kind, name };
            default:
                console.warn("Unknown term kind:", kind, entity);
                return null; // Or throw error
        }
    };

    /** Converts a Term object into Datascript transaction data (list of entity maps). Returns the transaction data and the db/id of the root term entity. */
    export const termToTransactionData = (term: Term): { tx: any[], rootId: any } => {
        const rootId = ds.tempid('term');
        let tx: any[] = [];
        const base = { ':db/id': rootId, ':term/kind': `:flowmind.term/${term.kind}` };

        switch (term.kind) {
            case 'struct':
                const argsData = (term.args ?? []).map(arg => termToTransactionData(arg));
                tx = argsData.flatMap(d => d.tx);
                tx.push({ ...base, ':term/name': term.name, ':term/args': argsData.map(d => d.rootId) });
                break;
            case 'list':
                const elementsData = (term.elements ?? []).map(el => termToTransactionData(el));
                tx = elementsData.flatMap(d => d.tx);
                tx.push({ ...base, ':term/elements': elementsData.map(d => d.rootId) });
                break;
            case 'atom':
            case 'var':
                tx.push({ ...base, ':term/name': term.name });
                break;
        }
        return { tx, rootId };
    };

    /** Pretty prints a term. */
    export const formatTerm = (term: Term | null): string => {
        if (!term) return 'null';
        switch (term.kind) {
            case 'atom': return term.name ?? 'nil';
            case 'var': return `?${term.name ?? 'unnamed'}`;
            case 'struct':
                const argsStr = (term.args ?? []).map(formatTerm).join(', ');
                return `${term.name ?? 'anon_struct'}(${argsStr})`;
            case 'list':
                const elsStr = (term.elements ?? []).map(formatTerm).join(', ');
                return `[${elsStr}]`;
            default: return 'invalid_term';
        }
    };

    /**
     * Attempts to unify two terms, returning the resulting bindings if successful.
     * Variables are represented as { kind: 'var', name: 'VarName' }.
     */
    export const unify = (
        t1: Term,
        t2: Term,
        bindings: Map<string, Term> = new Map()
    ): { success: boolean; bindings: Map<string, Term> } | null => {
        const resolve = (term: Term, currentBindings: Map<string, Term>): Term => {
            if (term.kind === 'var' && term.name && currentBindings.has(term.name)) {
                return resolve(currentBindings.get(term.name)!, currentBindings);
            }
            return term;
        };

        const term1 = resolve(t1, bindings);
        const term2 = resolve(t2, bindings);

        if (term1.kind === 'var' && term1.name) {
            if (term1.name === term2.name && term1.kind === term2.kind) return { success: true, bindings }; // Var equals itself
            const newBindings = new Map(bindings);
            newBindings.set(term1.name, term2);
            return { success: true, bindings: newBindings };
        }
        if (term2.kind === 'var' && term2.name) {
            // Already handled t1 being a var, so t1 is not a var here
            const newBindings = new Map(bindings);
            newBindings.set(term2.name, term1);
            return { success: true, bindings: newBindings };
        }

        if (term1.kind !== term2.kind) return null; // Different kinds

        if (term1.kind === 'atom') {
            return term1.name === term2.name ? { success: true, bindings } : null;
        }

        if (term1.kind === 'struct') {
            if (term1.name !== term2.name || term1.args?.length !== term2.args?.length) return null;
            let currentBindings = bindings;
            for (let i = 0; i < (term1.args?.length ?? 0); i++) {
                const result = unify(term1.args![i], term2.args![i], currentBindings);
                if (!result) return null;
                currentBindings = result.bindings;
            }
            return { success: true, bindings: currentBindings };
        }

        if (term1.kind === 'list') {
            if (term1.elements?.length !== term2.elements?.length) return null;
            let currentBindings = bindings;
            for (let i = 0; i < (term1.elements?.length ?? 0); i++) {
                const result = unify(term1.elements![i], term2.elements![i], currentBindings);
                if (!result) return null;
                currentBindings = result.bindings;
            }
            return { success: true, bindings: currentBindings };
        }

        console.error("Unhandled unification case:", term1, term2);
        return null; // Should not happen
    };

     /** Substitutes variables in a term based on the bindings map. */
     export const substitute = (term: Term, bindings: Map<string, Term>): Term => {
        if (term.kind === 'var' && term.name && bindings.has(term.name)) {
            // Recursively substitute in case the binding is another variable or complex term
            return substitute(bindings.get(term.name)!, bindings);
        }
        if (term.kind === 'struct' && term.args) {
            return { ...term, args: term.args.map(arg => substitute(arg, bindings)) };
        }
        if (term.kind === 'list' && term.elements) {
            return { ...term, elements: term.elements.map(el => substitute(el, bindings)) };
        }
        return term; // Atoms or unbound vars
    };
}


// ======== Database ========
namespace Database {
    import Term = Types.Term;
    import StatusValue = Types.StatusValue;

    /** Datascript schema definition. */
    const dbSchema = {
        // Common
        uuid: { ':db/unique': ':db.unique/identity', ':db/index': true, ':db/doc': 'Unique identifier for entities like Thoughts, Rules, Events' },
        kind: { ':db/index': true, ':db/doc': 'Entity type marker (e.g., :flowmind.kind/thought)' },
        createdAt: { ':db/valueType': ':db.type/instant', ':db/doc': 'Timestamp when the entity was created' },
        modifiedAt: { ':db/valueType': ':db.type/instant', ':db/doc': 'Timestamp when the entity was last modified' },

        // Term
        ':term/kind': { ':db/index': true, ':db/doc': 'Kind of term (:flowmind.term/atom, /var, /struct, /list)' },
        ':term/name': { ':db/index': true, ':db/doc': 'Name of atom, var, or struct' },
        ':term/args': { ':db/valueType': ':db.type/ref', ':db/cardinality': ':db.cardinality/many', ':db/doc': 'Arguments for struct term' },
        ':term/elements': { ':db/valueType': ':db.type/ref', ':db/cardinality': ':db.cardinality/many', ':db/doc': 'Elements for list term' },

        // Thought
        ':thought/content': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference to the Term entity representing the thought content' },
        ':thought/status': { ':db/index': true, ':db/doc': 'Processing status (:flowmind.status/pending, etc.)' },
        ':thought/truth': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference to truth value entity {pos, neg}' },
        ':thought/goal': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference to goal entity {value, source, time}' },
        ':thought/meta': { ':db/valueType': ':db.type/ref', ':db/cardinality': ':db.cardinality/many', ':db/doc': 'References to metadata map entries for the thought'},

        // Rule
        ':rule/head': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference to the Term entity for the rule head' },
        ':rule/body': { ':db/valueType': ':db.type/ref', ':db/cardinality': ':db.cardinality/many', ':db/doc': 'References to Term entities for the rule body operations' },
        ':rule/truth': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference to truth value entity for the rule' },
        ':rule/meta': { ':db/valueType': ':db.type/ref', ':db/cardinality': ':db.cardinality/many', ':db/doc': 'References to metadata map entries for the rule'},

        // Event
        ':event/type': { ':db/index': true, ':db/doc': 'Type of the event (e.g., :flowmind.event.type/tool_success)' },
        ':event/target_uuid': { ':db/index': true, ':db/doc': 'UUID of the entity this event primarily concerns' },
        ':event/data': { ':db/doc': 'Payload of the event (can be simple type or ref)' }, // Flexible type
        ':event/data_ref': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference to event data if it\'s complex' },

        // Meta Map Entry (used for :thought/meta and :rule/meta)
        ':meta_entry/key': { ':db/index': true, ':db/doc': 'Key of the metadata entry' },
        ':meta_entry/value_string': {':db/doc': 'String value for metadata'},
        ':meta_entry/value_ref': { ':db/valueType': ':db.type/ref', ':db/doc': 'Reference value for metadata'},
        ':meta_entry/value_boolean': { ':db/valueType': ':db.type/boolean', ':db/doc': 'Boolean value for metadata'},
        ':meta_entry/value_double': { ':db/valueType': ':db.type/double', ':db/doc': 'Number value for metadata'},
        ':meta_entry/value_instant': { ':db/valueType': ':db.type/instant', ':db/doc': 'Date value for metadata'},

         // Truth value component entity
         ':truth/pos': { ':db/valueType': ':db.type/long' },
         ':truth/neg': { ':db/valueType': ':db.type/long' },

         // Goal value component entity
         ':goal/value': { ':db/valueType': ':db.type/double' },
         ':goal/source': { ':db/index': true },
         ':goal/time': { ':db/valueType': ':db.type/instant' },

    };

    // Type helper for listener functions
    type DBListener = (db: DBInterface) => void;

    /** Provides an interface for interacting with the Datascript database. */
    export class DBInterface {
        private db: any;
        private history: any[][] = []; // Simple transaction history for undo
        private listeners: Set<DBListener> = new Set();

        constructor() {
            this.db = ds.empty_db(dbSchema);
        }

        /** Add a listener function to be called after transactions. */
        addListener(listener: DBListener) {
            this.listeners.add(listener);
        }

        /** Remove a listener function. */
        removeListener(listener: DBListener) {
            this.listeners.delete(listener);
        }

        /** Notify all listeners about a DB change. */
        private notifyListeners() {
            this.listeners.forEach(listener => listener(this));
        }

        /** Executes a transaction against the database. */
        transact(tx: any[]) {
            if (!tx || tx.length === 0) {
                return null;
            }
            // Add/update modifiedAt timestamp for relevant entities
            const now = new Date();
            const txWithTimestamps = tx.map(t => {
                if (t[':db/id'] && (typeof t[':db/id'] === 'number' && t[':db/id'] < 0) && t.kind?.startsWith(':flowmind.kind/')) {
                     // New entity, add createdAt
                     return { ...t, modifiedAt: now, createdAt: t.createdAt || now };
                } else if (t[':db/id'] && typeof t[':db/id'] === 'number' && t[':db/id'] > 0) {
                     // Existing entity, update modifiedAt
                    return { ...t, modifiedAt: now };
                } else if (t.uuid && typeof t[':db/id'] === 'string') {
                     // New entity with temp id string, add createdAt
                    return { ...t, modifiedAt: now, createdAt: t.createdAt || now };
                }
                // Add logic to find existing entities by uuid if needed for timestamp update
                return t;
            });


            try {
                const report = ds.transact(this.db, txWithTimestamps);
                if (report) {
                    this.db = report.db_after;
                    this.history.push(tx); // Store original tx for undo
                    if (this.history.length > 100) this.history.shift(); // Limit history size
                    this.notifyListeners(); // Notify UI or other components
                    return report;
                } else {
                     console.error('Transaction returned null report:', txWithTimestamps);
                     return null;
                }
            } catch (e: any) {
                console.error('Transaction failed:', e.message);
                console.error('Transaction data:', JSON.stringify(txWithTimestamps, null, 2));
                return null;
            }
        }

        /** Reverts the last transaction. */
        undo() {
            if (!this.history.length) return;
            this.history.pop(); // Remove the last transaction
            // Rebuild the database state from the remaining history
            let tempDb = ds.empty_db(dbSchema);
            this.history.forEach(tx => {
                try {
                    const report = ds.transact(tempDb, tx);
                    if (report) tempDb = report.db_after;
                } catch (e) {
                    console.error("Error replaying history during undo:", e);
                    // Decide how to handle errors here - potentially stop or skip problematic tx
                }
            });
            this.db = tempDb;
            this.notifyListeners();
        }

        /** Performs a Datascript query. */
        query(q: string, ...args: any[]): any[] {
            try {
                return ds.q(q, this.db, ...args);
            } catch (e: any) {
                console.error('Query failed:', e.message);
                console.error('Query:', q);
                console.error('Args:', args);
                return [];
            }
        }

        /** Pulls entity data based on an ID and pattern. */
        pull(id: number | string | any[], pattern: string): any | null {
            try {
                return ds.pull(this.db, pattern, id);
            } catch (e: any) {
                // Errors can happen if ID doesn't exist, treat as null
                // console.warn('Pull failed:', e.message, "ID:", id, "Pattern:", pattern);
                return null;
            }
        }

         /** Pulls multiple entities based on IDs and pattern. */
        pullMany(ids: (number | string | any[])[], pattern: string): (any | null)[] {
            try {
                return ds.pull_many(this.db, pattern, ids);
            } catch (e: any) {
                console.error('PullMany failed:', e.message);
                return ids.map(() => null);
            }
        }

        /** Finds a single entity ID based on attribute and value. */
        findEntityIdByAttribute(attribute: string, value: any): number | null {
            const result = this.query(`[:find ?e . :where [?e ${attribute} ?v] [(== ?v ${JSON.stringify(value)})]]`);
            return result ?? null; // ds.q returns entity id directly or null if not found
        }

        /** Exports the database state to a JSON string. */
        export(): string {
            try {
                return JSON.stringify(ds.db_to_json(this.db));
            } catch (e: any) {
                console.error('DB Export failed:', e.message);
                return "";
            }
        }

        /** Imports database state from a JSON string. */
        import(json: string): boolean {
            try {
                this.db = ds.json_to_db(JSON.parse(json), dbSchema);
                this.history = []; // Clear history after import
                this.notifyListeners();
                return true;
            } catch (e: any) {
                console.error('DB Import failed:', e.message);
                this.db = ds.empty_db(dbSchema); // Reset to empty on failure
                this.notifyListeners();
                return false;
            }
        }

        /** Helper to build metadata transaction data */
        static buildMetaTx(meta: Map<string, any>): { tx: any[], refs: any[] } {
            const metaTx: any[] = [];
            const metaRefs: any[] = [];
            for (const [key, value] of meta.entries()) {
                const metaId = ds.tempid('meta');
                metaRefs.push(metaId);
                const entry: any = { ':db/id': metaId, ':meta_entry/key': key };
                 if (typeof value === 'string') entry[':meta_entry/value_string'] = value;
                 else if (typeof value === 'number') entry[':meta_entry/value_double'] = value;
                 else if (typeof value === 'boolean') entry[':meta_entry/value_boolean'] = value;
                 else if (value instanceof Date) entry[':meta_entry/value_instant'] = value;
                 else if (value && typeof value === 'object' && value[':db/id']) entry[':meta_entry/value_ref'] = value; // Assume it's a ref
                 else {
                    console.warn(`Unsupported metadata type for key "${key}": ${typeof value}. Storing as string.`);
                    entry[':meta_entry/value_string'] = JSON.stringify(value);
                 }
                metaTx.push(entry);
            }
            return { tx: metaTx, refs: metaRefs };
        }

         /** Helper to parse metadata from pulled entities */
        static parseMeta(metaEntities: any[]): Map<string, any> {
            const metaMap = new Map<string, any>();
            if (!metaEntities) return metaMap;
            for (const entry of metaEntities) {
                const key = entry[':meta_entry/key'];
                if (!key) continue;
                const value = entry[':meta_entry/value_string']
                           ?? entry[':meta_entry/value_ref'] // Might need further pulling
                           ?? entry[':meta_entry/value_boolean']
                           ?? entry[':meta_entry/value_double']
                           ?? entry[':meta_entry/value_instant']
                           ?? undefined;
                metaMap.set(key, value);
            }
            return metaMap;
        }
    }
}

// ======== Tools ========
namespace Tools {
    import Term = Types.Term;
    import Tool = Types.Tool;
    import Config = Types.Config;
    import UUID = Types.UUID;

    /** Tool for interacting with the LLM (generation, embedding, etc.). */
    export class LLMTool implements Tool {
        private llm: BaseLanguageModel;
        private embeddings: Embeddings;

        constructor(config: Config) {
            this.llm = new ChatOllama({
                baseUrl: config.llmEndpoint,
                model: config.ollamaModel,
            });
             this.embeddings = new OllamaEmbeddings({
                 baseUrl: config.llmEndpoint,
                 model: config.ollamaModel,
             });
        }

        async execute(params: Term): Promise<Term> {
            if (params.kind !== 'struct' || !params.name || !params.args) return { kind: 'atom', name: 'error:invalid_params' };

            const operation = params.name;
            const args = params.args;

            try {
                switch (operation) {
                    case 'generate_text': // e.g., generate_text("prompt text")
                        if (args[0]?.kind === 'atom' && args[0].name) {
                            const prompt = args[0].name;
                            console.log(`LLMTool: Generating text for prompt: "${prompt.substring(0, 50)}..."`);
                            const response = await this.llm.invoke(prompt);
                            const responseText = typeof response.content === 'string' ? response.content : JSON.stringify(response.content);
                            console.log(`LLMTool: Generation result: "${responseText.substring(0, 50)}..."`);
                            return { kind: 'atom', name: responseText.trim() };
                        }
                        break;

                    case 'embed_text': // e.g., embed_text("text to embed")
                        if (args[0]?.kind === 'atom' && args[0].name) {
                            const text = args[0].name;
                            console.log(`LLMTool: Embedding text: "${text.substring(0,50)}..."`);
                            const embedding = await this.embeddings.embedQuery(text);
                            console.log(`LLMTool: Embedding result dimension: ${embedding.length}`);
                            return {
                                kind: 'list',
                                elements: embedding.map((n: number) => ({ kind: 'atom', name: n.toString() }))
                            };
                        }
                        break;

                    case 'synthesize_rule': // e.g., synthesize_rule("description of desired rule")
                         if (args[0]?.kind === 'atom' && args[0].name) {
                             const description = args[0].name;
                             const prompt = `Based on the following description, synthesize a FlowMind rule.
    A rule has a 'head' term and a 'body' list of operation terms.
    Operations start with 'op:'. Examples: op:add_thought(...), op:tool(...), op:set(...).
    Terms can be: atom(name), var(?Name), struct(name, arg1, ...), list([el1, ...]).
    Use standard FlowMind term syntax. Output ONLY the rule structure as JSON-like text.
    Description: "${description}"
Synthesized Rule:`;
                             console.log(`LLMTool: Synthesizing rule for: "${description.substring(0, 50)}..."`);
                             const response = await this.llm.invoke(prompt);
                             const ruleText = typeof response.content === 'string' ? response.content : JSON.stringify(response.content);
                             console.log(`LLMTool: Synthesized rule text: ${ruleText}`);
                             // TODO: Add robust parsing of the LLM output into a Rule structure.
                             // For now, just return the raw text for potential manual processing or logging.
                             return { kind: 'atom', name: ruleText.trim() };
                         }
                         break;

                    default:
                        console.warn("LLMTool: Unsupported operation:", operation);
                        return { kind: 'atom', name: `error:unsupported_llm_op:${operation}` };
                }
            } catch (e: any) {
                console.error('LLMTool Error:', e.message);
                return { kind: 'atom', name: `error:llm_failed:${e.message}` };
            }
            return { kind: 'atom', name: 'error:invalid_llm_args' };
        }
    }

    /** Tool for interacting with the vector memory store. */
    export class MemoryTool implements Tool {
        private vectorStore: MemoryVectorStore;
        private embeddings: Embeddings;

        constructor(config: Config) {
            // Requires embeddings to function
             this.embeddings = new OllamaEmbeddings({
                 baseUrl: config.llmEndpoint,
                 model: config.ollamaModel,
             });
             // Initialize the vector store with the embeddings
            this.vectorStore = new MemoryVectorStore(this.embeddings);
            console.log("MemoryTool: Initialized MemoryVectorStore.");
        }

        async execute(params: Term): Promise<Term> {
            if (params.kind !== 'struct' || !params.name || !params.args) return { kind: 'atom', name: 'error:invalid_params' };

            const operation = params.name;
            const args = params.args;

            try {
                switch (operation) {
                    case 'add_entry': // e.g., add_entry("unique_id", "content text", { meta_key: "meta_value" })
                        if (args[0]?.kind === 'atom' && args[0].name && // ID
                            args[1]?.kind === 'atom' && args[1].name) { // Content
                            const entryId = args[0].name;
                            const content = args[1].name;
                            // Optional metadata struct: struct(meta, kv(key1, val1), kv(key2, val2), ...)
                            let metadata: Record<string, any> = { id: entryId, content: content };
                            if (args[2]?.kind === 'struct' && args[2].name === 'meta' && args[2].args) {
                                args[2].args.forEach(kv => {
                                    if (kv.kind === 'struct' && kv.name === 'kv' && kv.args?.length === 2 && kv.args[0].kind === 'atom' && kv.args[1].kind === 'atom') {
                                        metadata[kv.args[0].name!] = kv.args[1].name;
                                    }
                                });
                            }

                            console.log(`MemoryTool: Adding entry id=${entryId}, content="${content.substring(0,30)}..."`);
                            // Embed the content and add to store
                            // Note: MemoryVectorStore expects Document[]
                            const doc = new Document({ pageContent: content, metadata: metadata });
                            await this.vectorStore.addDocuments([doc]);
                            console.log(`MemoryTool: Entry added successfully.`);
                            return { kind: 'atom', name: 'ok' };
                        }
                        break;

                    case 'search_similar': // e.g., search_similar("query text", 3)
                         if (args[0]?.kind === 'atom' && args[0].name) { // Query text
                             const queryText = args[0].name;
                             const k = (args[1]?.kind === 'atom' && parseInt(args[1].name!, 10)) || 3; // Number of results
                             console.log(`MemoryTool: Searching for ${k} entries similar to "${queryText.substring(0, 30)}..."`);
                             const results = await this.vectorStore.similaritySearchWithScore(queryText, k);
                             console.log(`MemoryTool: Found ${results.length} results.`);
                              // Return a list of structs: [ struct(result, atom(content1), atom(score1)), ... ]
                              return {
                                  kind: 'list',
                                  elements: results.map(([doc, score]) => ({
                                      kind: 'struct',
                                      name: 'search_result',
                                      args: [
                                          { kind: 'atom', name: doc.pageContent },
                                          { kind: 'atom', name: score.toFixed(4) },
                                           // Could add ID or other metadata here if needed
                                           { kind: 'atom', name: doc.metadata?.id || 'unknown_id'}
                                      ]
                                  }))
                              };
                         }
                         break;

                    default:
                        console.warn("MemoryTool: Unsupported operation:", operation);
                        return { kind: 'atom', name: `error:unsupported_memory_op:${operation}` };
                }
            } catch (e: any) {
                console.error('MemoryTool Error:', e.message);
                return { kind: 'atom', name: `error:memory_failed:${e.message}` };
            }
             return { kind: 'atom', name: 'error:invalid_memory_args' };
        }
    }

    /** Tool for requesting input from the user. */
    export class UserInteractionTool implements Tool {
         // This tool no longer needs direct UI coupling.
         // It signals the need for input by creating/updating thoughts.
        constructor(private db: Database.DBInterface) {}

        async execute(params: Term): Promise<Term> {
             // Expected format: prompt("Prompt text", ?VarToBindResponseTo)
             // Or just prompt("Prompt text") if response isn't needed directly by rule
            if (params.kind !== 'struct' || params.name !== 'prompt' || !params.args || params.args.length === 0 || params.args[0].kind !== 'atom' || !params.args[0].name) {
                return { kind: 'atom', name: 'error:invalid_prompt_params' };
            }

            const promptText = params.args[0].name;
            const responseVar = params.args[1]?.kind === 'var' ? params.args[1].name : undefined;
            const promptThoughtUUID = uuid();

            console.log(`UserInteractionTool: Requesting prompt: "${promptText}"`);

            // Create a new thought representing the prompt itself
            // This thought will be picked up by the UI
            const promptTextTermData = TermUtils.termToTransactionData({ kind: 'atom', name: promptText });
            const promptMeta = new Map<string, any>([
                [META_KEYS.UI_PROMPT_TEXT, promptText],
                // If a response is needed, mark this thought as waiting for it.
                // The action handler will later look for a response thought referencing this UUID.
                 ...(responseVar ? [[META_KEYS.WAITING_FOR_USER_INPUT, true]] : [])
            ]);
             const metaTxData = Database.DBInterface.buildMetaTx(promptMeta);

             const promptThoughtTx = [
                ...promptTextTermData.tx,
                ...metaTxData.tx,
                 {
                     ':db/id': ds.tempid('thought'),
                     uuid: promptThoughtUUID,
                     kind: ':flowmind.kind/thought',
                     ':thought/content': promptTextTermData.rootId,
                     ':thought/status': Types.Status.WAITING, // Waiting for user interaction
                     ':thought/truth': { ':db/id': ds.tempid('truth'), ':truth/pos': 1, ':truth/neg': 0 },
                     ':thought/goal': { ':db/id': ds.tempid('goal'), ':goal/value': 0.95, ':goal/source': 'ui_prompt_tool', ':goal/time': new Date() },
                     ':thought/meta': metaTxData.refs,
                     createdAt: new Date(),
                     modifiedAt: new Date(),
                 }
             ];

             this.db.transact(promptThoughtTx);

            // The tool itself returns the UUID of the prompt thought.
            // The rule that called this might use this UUID later to find the response.
            return { kind: 'atom', name: promptThoughtUUID };
        }
    }

    // Tool for suggesting goals (Example using LLM)
    export class GoalProposalTool implements Tool {
        private llmTool: LLMTool;
        constructor(config: Config) {
            this.llmTool = new LLMTool(config); // Reuse LLMTool for generation
        }

        async execute(params: Term): Promise<Term> {
             // e.g., suggest_goals("context text")
             if (params.kind !== 'struct' || params.name !== 'suggest_goals' || !params.args || args[0].kind !== 'atom' || !args[0].name) {
                return { kind: 'atom', name: 'error:invalid_goal_suggestion_params' };
            }
            const context = params.args[0].name;
            const prompt = `Based on the following context, suggest one or two potential goals or next steps, formatted as a list term: [atom("goal 1"), atom("goal 2")]. Context: "${context}" Suggested Goals:`;
            console.log(`GoalProposalTool: Suggesting goals based on: "${context.substring(0,50)}..."`);

            // Use the LLM tool to generate suggestions
            const llmResult = await this.llmTool.execute({
                kind: 'struct',
                name: 'generate_text',
                args: [{ kind: 'atom', name: prompt }]
            });

            if (llmResult.kind === 'atom' && llmResult.name && !llmResult.name.startsWith('error:')) {
                 // Basic parsing attempt - LLM should ideally return the list structure
                 // This is fragile and needs improvement or stricter LLM prompting.
                 try {
                    // Try to match atom("...") pattern
                    const matches = llmResult.name.match(/atom\("([^"]+)"\)/g);
                    if (matches) {
                         const goals = matches.map(m => m.match(/atom\("([^"]+)"\)/)![1]);
                         console.log(`GoalProposalTool: Suggested goals:`, goals);
                         return {
                             kind: 'list',
                             elements: goals.map(g => ({ kind: 'atom', name: g }))
                         };
                    }
                 } catch (parseError) {
                     console.error("GoalProposalTool: Failed to parse LLM response for goals:", parseError);
                     return { kind: 'atom', name: 'error:goal_suggestion_parsing_failed' };
                 }
            }

            console.error("GoalProposalTool: LLM failed to generate goal suggestions.");
            return { kind: 'atom', name: 'error:goal_suggestion_failed' };
        }
    }

    // Placeholder/Commented Collaboration Tool
    /*
    export class CollaborationTool implements Tool {
        private socket: Socket | null = null;
        constructor() {
            try {
                // Replace with actual server endpoint and connection logic
                this.socket = io('http://localhost:3001', { autoConnect: false }); // Example endpoint
                this.socket.connect();
                console.log("CollaborationTool: Attempting to connect...");
                this.socket.on('connect', () => console.log("CollaborationTool: Connected."));
                this.socket.on('disconnect', () => console.log("CollaborationTool: Disconnected."));
                this.socket.on('connect_error', (err) => console.error("CollaborationTool: Connection Error:", err.message));
            } catch (e: any) {
                console.error("CollaborationTool: Failed to initialize socket.io:", e.message);
            }
        }

        async execute(params: Term): Promise<Term> {
            if (!this.socket || !this.socket.connected) {
                return { kind: 'atom', name: 'error:collaboration_disconnected' };
            }
            // e.g., send_thought("target_user_or_channel", struct(thought_content))
            if (params.kind !== 'struct' || params.name !== 'send_thought' || !params.args || args.length < 2 || args[0].kind !== 'atom' || !args[0].name) {
                return { kind: 'atom', name: 'error:invalid_collaboration_params' };
            }

            const target = params.args[0].name;
            const thoughtTerm = params.args[1]; // Pass the term structure directly

            try {
                // TODO: Need robust serialization of the Term object for sending
                const serializedThought = JSON.stringify(thoughtTerm); // Basic serialization
                console.log(`CollaborationTool: Sending thought to ${target}`);
                this.socket.emit('share_thought', { target, thought: serializedThought });
                return { kind: 'atom', name: 'ok' };
            } catch (e: any) {
                console.error('CollaborationTool Error:', e.message);
                return { kind: 'atom', name: `error:collaboration_failed:${e.message}` };
            }
        }
    }
    */

    /** Registry to hold and access available tools. */
    export class ToolRegistry {
        private tools: Map<string, Tool>;

        constructor(config: Config, db: Database.DBInterface) {
            this.tools = new Map<string, Tool>([
                ['llm', new LLMTool(config)],
                ['memory', new MemoryTool(config)], // Memory tool requires config for embeddings
                ['user_interaction', new UserInteractionTool(db)],
                ['goal_proposal', new GoalProposalTool(config)],
                // ['collaboration', new CollaborationTool()], // Enable if using
            ]);
            console.log("ToolRegistry: Initialized with tools:", Array.from(this.tools.keys()));
        }

        get(name: string): Tool | undefined {
            return this.tools.get(name);
        }
    }
}

// ======== Engine ========
namespace Engine {
    import Term = Types.Term;
    import Rule = Types.Rule;
    import Action = Types.Action;
    import Thought = Types.Thought;
    import Status = Types.Status;
    import DBInterface = Database.DBInterface;

    /** The core inference engine that processes thoughts based on rules. */
    export class InferenceEngine {
        constructor(private db: DBInterface) {}

        /**
         * Processes a single thought, finding matching rules and generating
         * transactions and actions based on the selected rule.
         * @param triggerThoughtUUID The UUID of the thought to process (must be ACTIVE).
         * @returns An object containing transaction data and actions, or null if processing fails.
         */
        infer(triggerThoughtUUID: Types.UUID): { transactionData: any[]; actions: Action[] } | null {
            const thoughtEid = this.db.findEntityIdByAttribute('uuid', triggerThoughtUUID);
            if (!thoughtEid) {
                 console.warn(`Engine.infer: Thought not found for UUID: ${triggerThoughtUUID}`);
                 return null; // Thought disappeared?
            }

            const thought = this.db.pull(thoughtEid, [
                'uuid',
                ':thought/status',
                { ':thought/content': ['*'] }, // Pull content entity directly
                {':thought/truth': ['*']},
                 {':thought/goal': ['*']},
                 {':thought/meta': [{':meta_entry/key': {}}]} // Need keys to check for things like prompts
            ]);

            if (!thought || thought[':thought/status'] !== Status.ACTIVE) {
                console.warn(`Engine.infer: Thought ${triggerThoughtUUID} not found or not active.`);
                 // Maybe revert status back to pending? Or log error?
                 // For now, just return null indicating no inference happened.
                return null;
            }
             // Don't process thoughts waiting for user input further until response arrives
             const isWaitingForUser = Database.DBInterface.parseMeta(thought[':thought/meta'] ?? []).get(META_KEYS.WAITING_FOR_USER_INPUT);
             if (isWaitingForUser) {
                 console.log(`Engine.infer: Skipping thought ${triggerThoughtUUID} as it's waiting for user input.`);
// Change status back to WAITING if it somehow became ACTIVE
return { transactionData: [{':db/id': thoughtEid, ':thought/status': Status.WAITING}], actions: []};
}

const contentTerm = TermUtils.entityToTerm(this.db, thought[':thought/content']?.[':db/id']);
if (!contentTerm) {
    console.error(`Engine.infer: Failed to decode term content for thought ${triggerThoughtUUID}.`);
    return {
        transactionData: [
            { ':db/id': thoughtEid, ':thought/status': Status.FAILED },
            this.createEventTx(triggerThoughtUUID, ':flowmind.event.type/content_decode_error', 'Failed to decode content term')
        ],
        actions: []
    };
}
console.log(`Engine.infer: Processing thought ${triggerThoughtUUID}: ${TermUtils.formatTerm(contentTerm)}`);

// Find potentially matching rules
// Query for all rules, then filter. Could be optimized if rules indexed by head structure.
const allRuleEids = this.db.query('[:find ?r :where [?r :kind :flowmind.kind/rule]]').map((r: any) => r[0]);
const allRules = this.db.pullMany(allRuleEids, [
    'uuid',
    { ':rule/head': ['*'] },
    { ':rule/body': ['*'] }, // Pull body op terms
    { ':rule/truth': ['*'] },
    'createdAt' // For potential tie-breaking or recency bias
]);

const matchingRules = allRules.map((r, idx) => ({ rule: r, eid: allRuleEids[idx] }))
    .filter(item => item.rule) // Filter out nulls if pullMany failed
    .map(item => {
        const headTerm = TermUtils.entityToTerm(this.db, item.rule[':rule/head']?.[':db/id']);
        if (!headTerm) return null; // Skip rules with invalid heads
        const unificationResult = TermUtils.unify(contentTerm, headTerm);
        return unificationResult ? { ...item, bindings: unificationResult.bindings, headTerm } : null;
    })
    .filter(Boolean); // Remove nulls (non-matching or invalid rules)

if (matchingRules.length === 0) {
    console.log(`Engine.infer: No rule matched thought ${triggerThoughtUUID}: ${TermUtils.formatTerm(contentTerm)}`);
    return {
        transactionData: [
            // Optionally set status to failed or keep active for rule synthesis? Let's mark failed for now.
            { ':db/id': thoughtEid, ':thought/status': Status.FAILED },
            this.createEventTx(triggerThoughtUUID, ':flowmind.event.type/no_rule_match', TermUtils.formatTerm(contentTerm))
        ],
        actions: [{ // Trigger rule synthesis or logging
            type: 'log', // Or potentially a tool call to synthesize
            context: { triggerUUID: triggerThoughtUUID },
            params: { kind: 'struct', name: 'log_message', args: [{ kind: 'atom', name: `No rule match for ${TermUtils.formatTerm(contentTerm)}` }] }
        }]
    };
}

// Select a rule (using probabilistic sampling based on truth value)
const selectedMatch = this.selectRule(matchingRules);
const { rule: selectedRule, bindings, eid: selectedRuleEid } = selectedMatch;
const ruleUUID = selectedRule.uuid;
console.log(`Engine.infer: Selected rule ${ruleUUID} for thought ${triggerThoughtUUID}`);

// Decode body operation terms
const bodyOpTermIds = selectedRule[':rule/body']?.map((ref: any) => ref[':db/id']) || [];
const bodyOpTerms = bodyOpTermIds.map((id: number) => TermUtils.entityToTerm(this.db, id)).filter(Boolean) as Term[];

// Execute operations defined in the rule body
const { tx: opTx, actions: opActions } = this.executeOps(bodyOpTerms, bindings, triggerThoughtUUID, ruleUUID);

// Update triggering thought status (usually to done, unless rule specifies otherwise)
const finalStatus = opTx.some(t => t[':db/id'] === thoughtEid && t[':thought/status'])
    ? null // Status already set by an op:set
    : { ':db/id': thoughtEid, ':thought/status': Status.DONE };

const finalTx = [
    ...opTx,
    ...(finalStatus ? [finalStatus] : []),
    this.createEventTx(triggerThoughtUUID, ':flowmind.event.type/rule_executed', { ruleUUID: ruleUUID, bindings: Object.fromEntries(bindings.entries()) }) // Log execution
];

// Feedback on rule truth value (optional) - simplified: assume success for now
const ruleTruth = selectedRule[':rule/truth'];
if (ruleTruth) {
    finalTx.push({ ':db/id': ruleTruth[':db/id'], ':truth/pos': (ruleTruth[':truth/pos'] ?? 0) + 1 });
} else {
    // Create truth value if missing?
    const truthId = ds.tempid('truth');
    finalTx.push({ ':db/id': truthId, ':truth/pos': 1, ':truth/neg': 0 });
    finalTx.push({ ':db/id': selectedRuleEid, ':rule/truth': truthId });
}


return {
    transactionData: finalTx,
    actions: opActions,
};
}

/** Selects a rule from matches based on truth values using softmax sampling. */
private selectRule(matches: any[]): any {
    if (matches.length === 1) return matches[0];

    // Calculate weights (e.g., confidence = pos / (pos + neg + smoothing))
    const weights = matches.map(match => {
        const truth = match.rule[':rule/truth'];
        const pos = truth?.[':truth/pos'] ?? 0;
        const neg = truth?.[':truth/neg'] ?? 0;
        return (pos + 1) / (pos + neg + 2); // Laplace smoothing
    });

    // Softmax sampling
    const totalExp = weights.reduce((sum, w) => sum + Math.exp(w), 0);
    if (totalExp === 0 || !isFinite(totalExp)) { // Avoid division by zero or NaN
        console.warn("Engine.selectRule: Non-finite softmax total, selecting uniformly.");
        return matches[Math.floor(Math.random() * matches.length)];
    }
    const probabilities = weights.map(w => Math.exp(w) / totalExp);

    let random = Math.random();
    for (let i = 0; i < probabilities.length; i++) {
        random -= probabilities[i];
        if (random <= 0) {
            return matches[i];
        }
    }
    return matches[matches.length - 1]; // Fallback
}

/** Executes operations defined in a rule body. */
private executeOps(
    ops: Term[],
    bindings: Map<string, Term>,
    triggerUUID: Types.UUID,
    ruleUUID: Types.UUID
): { tx: any[]; actions: Action[] } {
    const tx: any[] = [];
    const actions: Action[] = [];
    let currentBindings = new Map(bindings); // Allow ops to potentially add bindings (e.g., tool results)

    for (const opTerm of ops) {
        // Substitute variables in the operation term itself *before* processing
        const op = TermUtils.substitute(opTerm, currentBindings);

        if (op.kind !== 'struct' || !op.name || !op.args) {
            console.warn(`Engine.executeOps: Skipping invalid operation term: ${TermUtils.formatTerm(op)}`);
            continue;
        }

        // --- op:tool ---
        // Format: op:tool(atom(ToolName), struct(ToolParams), optional_var(?ResultVar))
        if (op.name === 'op:tool' && op.args.length >= 2 && op.args[0].kind === 'atom' && op.args[0].name && op.args[1].kind === 'struct') {
            const toolName = op.args[0].name;
            const toolParams = op.args[1];
            const resultVar = op.args[2]?.kind === 'var' ? op.args[2].name : undefined;

            console.log(`Engine.executeOps: Enqueuing tool action: ${toolName}(${TermUtils.formatTerm(toolParams)}) ${resultVar ? `-> ?${resultVar}` : ''}`);
            actions.push({
                type: 'tool',
                name: toolName,
                params: toolParams,
                context: {
                    triggerUUID,
                    ruleUUID,
                    resultBindingVar: resultVar,
                    // If resultVar exists, the action handler needs to know which thought
                    // should receive the result. This is tricky. Let's assume for now
                    // the *triggering* thought might be updated or a *new* thought created.
                    // We might need a dedicated 'waiting' thought created by the rule.
                    // Let's pass triggerUUID, handler logic will need refinement.
                    waitingThoughtUUID: resultVar ? triggerUUID : undefined // Needs refinement - maybe pass a dedicated waiter UUID?
                }
            });
            // If the tool needs to bind a result, the triggering thought might pause
            if (resultVar) {
                // Mark the *triggering* thought as waiting. This might not always be correct.
                // A better pattern is for the rule to create a *new* thought that waits.
                tx.push({ ':db/id': this.db.findEntityIdByAttribute('uuid', triggerUUID), ':thought/status': Status.WAITING });
            }
        }
            // --- op:add_thought ---
        // Format: op:add_thought(struct(Content), struct(Goal), optional_struct(Meta))
        else if (op.name === 'op:add_thought' && op.args.length >= 1) {
            const contentTerm = op.args[0];
            const goalTerm = op.args[1]; // Optional: struct(goal, atom(Value), atom(Source))
            const metaTerm = op.args[2]; // Optional: struct(meta, kv(K1,V1), kv(K2,V2), ...)

            const newThoughtUUID = uuid();
            console.log(`Engine.executeOps: Adding new thought (${newThoughtUUID.substring(0, 8)}...): ${TermUtils.formatTerm(contentTerm)}`);

            // Content Term
            const contentTxData = TermUtils.termToTransactionData(contentTerm);
            tx.push(...contentTxData.tx);

            // Goal Term (defaults if not provided)
            const goalValue = (goalTerm?.kind === 'struct' && goalTerm.args?.[0]?.kind === 'atom' && parseFloat(goalTerm.args[0].name!)) || 0.5;
            const goalSource = (goalTerm?.kind === 'struct' && goalTerm.args?.[1]?.kind === 'atom' && goalTerm.args[1].name) || `rule:${ruleUUID.substring(0, 8)}`;
            const goalId = ds.tempid('goal');
            tx.push({ ':db/id': goalId, ':goal/value': goalValue, ':goal/source': goalSource, ':goal/time': new Date() });

            // Meta Term
            const baseMeta = new Map<string, any>([
                [META_KEYS.PARENT_UUID, triggerUUID], // Link to triggering thought
                [META_KEYS.SOURCE_RULE_UUID, ruleUUID]
            ]);
            // Add meta from rule, if provided
            if (metaTerm?.kind === 'struct' && metaTerm.name === 'meta' && metaTerm.args) {
                metaTerm.args.forEach(kv => {
                    if (kv.kind === 'struct' && kv.name === 'kv' && kv.args?.length === 2 && kv.args[0].kind === 'atom' && kv.args[0].name) {
                        // Value could be atom or var - vars should have been substituted
                        const metaValue = kv.args[1].kind === 'atom' ? kv.args[1].name
                            : kv.args[1].kind === 'list' ? TermUtils.formatTerm(kv.args[1]) // Basic stringify for lists
                                : 'unsupported_meta_value';
                        baseMeta.set(kv.args[0].name, metaValue);
                    }
                });
            }
            const metaTxData = Database.DBInterface.buildMetaTx(baseMeta);
            tx.push(...metaTxData.tx);

            // Truth Term (default)
            const truthId = ds.tempid('truth');
            tx.push({ ':db/id': truthId, ':truth/pos': 0, ':truth/neg': 0 });

            // Thought Entity
            tx.push({
                ':db/id': ds.tempid('thought'),
                uuid: newThoughtUUID,
                kind: ':flowmind.kind/thought',
                ':thought/content': contentTxData.rootId,
                ':thought/status': Status.PENDING, // New thoughts start pending
                ':thought/truth': truthId,
                ':thought/goal': goalId,
                ':thought/meta': metaTxData.refs,
                createdAt: new Date(),
                modifiedAt: new Date(),
            });
        }
            // --- op:set ---
        // Format: op:set(atom(TargetUUID), atom(:attribute/name), ValueTerm)
        else if (op.name === 'op:set' && op.args.length === 3 && op.args[0].kind === 'atom' && op.args[0].name && op.args[1].kind === 'atom' && op.args[1].name) {
            const targetUUID = op.args[0].name;
            const attribute = op.args[1].name; // e.g., :thought/status
            const valueTerm = op.args[2];

            console.log(`Engine.executeOps: Setting ${attribute} on ${targetUUID} to ${TermUtils.formatTerm(valueTerm)}`);

            const targetEid = this.db.findEntityIdByAttribute('uuid', targetUUID);
            if (targetEid) {
                // Convert valueTerm to appropriate DB value
                let dbValue: any;
                if (valueTerm.kind === 'atom') {
                    dbValue = valueTerm.name;
                    // Handle specific types like status enums
                    if (attribute === ':thought/status' && Object.values(Types.Status).includes(dbValue as Types.StatusValue)) {
                        // Valid status
                    } else if (attribute === ':thought/status') {
                        console.warn(`Engine.executeOps: Invalid status value for op:set: ${dbValue}`);
                        dbValue = undefined; // Prevent setting invalid status
                    }
                } else if (valueTerm.kind === 'list' || valueTerm.kind === 'struct') {
                    // Setting complex values requires creating/linking entities
                    const valueTxData = TermUtils.termToTransactionData(valueTerm);
                    tx.push(...valueTxData.tx);
                    // Assume the attribute expects a reference
                    if (attribute.endsWith('/content') || attribute.endsWith('/head') || attribute.endsWith('/body') || attribute.endsWith('/ref')) {
                        dbValue = valueTxData.rootId;
                    } else {
                        console.warn(`Engine.executeOps: Setting complex term to non-ref attribute "${attribute}" might not work as expected.`);
                        dbValue = TermUtils.formatTerm(valueTerm); // Fallback to string representation?
                    }
                } else {
                    console.warn(`Engine.executeOps: Cannot set attribute ${attribute} using value term of kind ${valueTerm.kind}`);
                    dbValue = undefined;
                }

                if (dbValue !== undefined) {
                    tx.push({ ':db/id': targetEid, [attribute]: dbValue });
                }
            } else {
                console.warn(`Engine.executeOps: Target entity ${targetUUID} not found for op:set.`);
                // Maybe create an event?
                tx.push(this.createEventTx(triggerUUID, ':flowmind.event.type/op_set_failed', { targetUUID: targetUUID, reason: 'Not found' }));
            }
        }
            // --- op:log ---
        // Format: op:log(TermToLog)
        else if (op.name === 'op:log' && op.args.length === 1) {
            const logTerm = op.args[0];
            const message = TermUtils.formatTerm(logTerm);
            console.log(`Engine.executeOps (Rule ${ruleUUID.substring(0,8)}): ${message}`);
            // Also create a log event in the DB
            tx.push(this.createEventTx(triggerUUID, ':flowmind.event.type/log_message', message));
            // Could also add a log action for external handling
            actions.push({ type: 'log', context: { triggerUUID, ruleUUID }, params: logTerm });
        }
        else {
            console.warn(`Engine.executeOps: Unrecognized operation in rule ${ruleUUID}: ${op.name}`);
            tx.push(this.createEventTx(triggerUUID, ':flowmind.event.type/unknown_operation', { ruleUUID: ruleUUID, opName: op.name }));
        }
    }

    return { tx, actions };
}

/** Helper to create a transaction map for a new event */
private createEventTx(targetUUID: Types.UUID | null, type: string, data: any): any {
    const eventUUID = uuid();
    const eventData = (typeof data === 'string' || typeof data === 'number' || typeof data === 'boolean' || data instanceof Date)
        ? data
        : JSON.stringify(data); // Basic serialization for complex data

    return {
        ':db/id': ds.tempid('event'),
        uuid: eventUUID,
        kind: ':flowmind.kind/event',
        ':event/type': type,
        ':event/target_uuid': targetUUID, // May be null if event is global
        ':event/data': eventData, // Store simple data directly
        // ':event/data_ref': null, // Use if data needs to be an entity reference
        time: new Date(),
        createdAt: new Date(),
        modifiedAt: new Date(),
    };
}
}
}

// ======== Action Handler ========
namespace ActionHandler {
    import Action = Types.Action;
    import Term = Types.Term;
    import ToolRegistry = Tools.ToolRegistry;
    import DBInterface = Database.DBInterface;
    import Status = Types.Status;

    /** Handles executing actions generated by the InferenceEngine, such as tool calls. */
    export class ActionHandler {
        private queue: Action[] = [];
        private processing: boolean = false;

        constructor(private db: DBInterface, private tools: ToolRegistry) {}

        /** Adds actions to the processing queue. */
        enqueue(actions: Action[]) {
            if (!actions || actions.length === 0) return;
            this.queue.push(...actions);
            this.processQueue(); // Start processing if not already active
        }

        /** Processes the action queue asynchronously. */
        private async processQueue() {
            if (this.processing) return; // Already processing
            this.processing = true;

            while (this.queue.length > 0) {
                const action = this.queue.shift()!;
                console.log(`ActionHandler: Processing action type=${action.type} name=${action.name || 'N/A'} trigger=${action.context.triggerUUID.substring(0,8)}`);

                try {
                    switch (action.type) {
                        case 'tool':
                            await this.handleToolAction(action);
                            break;
                        case 'memory':
                            // TODO: Implement direct memory actions if needed (separate from MemoryTool)
                            console.warn("ActionHandler: 'memory' action type not fully implemented.");
                            this.logEvent(action.context.triggerUUID, ':flowmind.event.type/action_unsupported', { type: action.type });
                            break;
                        case 'log':
                            // Handled directly by engine rule for now, could add external logging here
                            console.log(`ActionHandler (Log): Trigger=${action.context.triggerUUID.substring(0,8)}, Rule=${action.context.ruleUUID?.substring(0,8)}, Message=${TermUtils.formatTerm(action.params ?? null)}`);
                            this.logEvent(action.context.triggerUUID, ':flowmind.event.type/log_action_processed', { message: TermUtils.formatTerm(action.params ?? null) });
                            break;
                        default:
                            console.warn(`ActionHandler: Unknown action type: ${(action as any).type}`);
                            this.logEvent(action.context.triggerUUID, ':flowmind.event.type/action_unknown', { type: (action as any).type });
                    }
                } catch (error: any) {
                    console.error(`ActionHandler: Error processing action: ${error.message}`, action);
                    this.logEvent(action.context.triggerUUID, ':flowmind.event.type/action_handler_error', { actionType: action.type, error: error.message });

                    // If a tool failed, potentially mark the waiting thought as failed
                    if (action.type === 'tool' && action.context.waitingThoughtUUID) {
                        const waitingEid = this.db.findEntityIdByAttribute('uuid', action.context.waitingThoughtUUID);
                        if (waitingEid) {
                            this.db.transact([{ ':db/id': waitingEid, ':thought/status': Status.FAILED, modifiedAt: new Date() }]);
                        }
                    }
                }
            }

            this.processing = false;
        }

        /** Handles the execution of a 'tool' action. */
        private async handleToolAction(action: Action) {
            if (!action.name || !action.params) {
                console.error("ActionHandler: Invalid 'tool' action - missing name or params.", action);
                this.logEvent(action.context.triggerUUID, ':flowmind.event.type/tool_action_invalid', { reason: 'Missing name or params' });
                return;
            }

            const tool = this.tools.get(action.name);
            if (!tool) {
                console.error(`ActionHandler: Tool "${action.name}" not found.`);
                this.logEvent(action.context.triggerUUID, ':flowmind.event.type/tool_not_found', { toolName: action.name });
                // Mark waiting thought as failed if applicable
                if (action.context.waitingThoughtUUID) {
                    const waitingEid = this.db.findEntityIdByAttribute('uuid', action.context.waitingThoughtUUID);
                    if (waitingEid) this.db.transact([{ ':db/id': waitingEid, ':thought/status': Status.FAILED }]);
                }
                return;
            }

            console.log(`ActionHandler: Executing tool "${action.name}"...`);
            const resultTerm = await tool.execute(action.params);
            console.log(`ActionHandler: Tool "${action.name}" result: ${TermUtils.formatTerm(resultTerm)}`);

            this.logEvent(action.context.triggerUUID, ':flowmind.event.type/tool_success', {
                toolName: action.name,
                result: TermUtils.formatTerm(resultTerm) // Log simplified result
            });

            // --- Handle Tool Result ---
            // If resultBindingVar is set, we need to update the thought that was waiting.
            // This logic assumes the rule engine set the *triggering* thought to WAITING.
            // A more robust system might involve the rule creating a dedicated "waiter" thought.
            if (action.context.resultBindingVar && action.context.waitingThoughtUUID) {
                const waitingThoughtEid = this.db.findEntityIdByAttribute('uuid', action.context.waitingThoughtUUID);
                if (waitingThoughtEid) {
                    console.log(`ActionHandler: Binding result to var "?${action.context.resultBindingVar}" for waiting thought ${action.context.waitingThoughtUUID.substring(0,8)}`);

                    // Create a new thought representing the result, linking it back.
                    const resultThoughtUUID = uuid();
                    const resultTxData = TermUtils.termToTransactionData(resultTerm);
                    const resultMeta = new Map<string, any>([
                        [META_KEYS.TOOL_RESULT_FOR, action.context.waitingThoughtUUID],
                        [META_KEYS.SOURCE_RULE_UUID, action.context.ruleUUID]
                    ]);
                    const metaTxData = Database.DBInterface.buildMetaTx(resultMeta);

                    const newThoughtTx = [
                        ...resultTxData.tx,
                        ...metaTxData.tx,
                        {
                            ':db/id': ds.tempid('thought'),
                            uuid: resultThoughtUUID,
                            kind: ':flowmind.kind/thought',
                            ':thought/content': resultTxData.rootId,
                            // This result thought itself might trigger further rules
                            ':thought/status': Status.PENDING,
                            ':thought/truth': { ':db/id': ds.tempid('truth'), ':truth/pos': 1, ':truth/neg': 0 }, // Result is initially trusted
                            // Goal value could be inherited or set based on context
                            ':thought/goal': { ':db/id': ds.tempid('goal'), ':goal/value': 0.6, ':goal/source': 'tool_result', ':goal/time': new Date() },
                            ':thought/meta': metaTxData.refs,
                            createdAt: new Date(),
                            modifiedAt: new Date(),
                        },
                        // Crucially, reactivate the waiting thought so the engine can potentially process the result.
                        // Alternatively, a specific rule could look for thoughts with META_KEYS.TOOL_RESULT_FOR.
                        // Reactivating seems simpler for now.
                        { ':db/id': waitingThoughtEid, ':thought/status': Status.PENDING, modifiedAt: new Date() }
                    ];
                    this.db.transact(newThoughtTx);

                } else {
                    console.warn(`ActionHandler: Waiting thought ${action.context.waitingThoughtUUID} not found to bind result.`);
                    this.logEvent(action.context.triggerUUID, ':flowmind.event.type/result_bind_failed', { waitingUUID: action.context.waitingThoughtUUID, reason: 'Not found' });
                }
            } else if (resultTerm.name?.startsWith('error:')) {
                // If the tool returned an error, mark the waiting thought as failed
                if (action.context.waitingThoughtUUID) {
                    const waitingEid = this.db.findEntityIdByAttribute('uuid', action.context.waitingThoughtUUID);
                    if (waitingEid) {
                        this.db.transact([{ ':db/id': waitingEid, ':thought/status': Status.FAILED, modifiedAt: new Date() }]);
                        this.logEvent(action.context.waitingThoughtUUID, ':flowmind.event.type/thought_failed_due_to_tool', { toolName: action.name, error: resultTerm.name });
                    }
                }
            }
        }

        /** Helper to log an event to the database */
        private logEvent(targetUUID: Types.UUID | null, type: string, data: any) {
            const eventUUID = uuid();
            const eventData = (typeof data === 'string' || typeof data === 'number' || typeof data === 'boolean' || data instanceof Date)
                ? data
                : JSON.stringify(data); // Basic serialization

            this.db.transact([{
                ':db/id': ds.tempid('event'),
                uuid: eventUUID,
                kind: ':flowmind.kind/event',
                ':event/type': type,
                ':event/target_uuid': targetUUID,
                ':event/data': eventData,
                time: new Date(),
                createdAt: new Date(),
                modifiedAt: new Date(),
            }]);
        }
    }
}

// ======== Scheduler ========
namespace Scheduler {
    import DBInterface = Database.DBInterface;
    import InferenceEngine = Engine.InferenceEngine;
    import ActionHandler = ActionHandler.ActionHandler;
    import Status = Types.Status;

    /** Manages the execution flow, selecting thoughts to process. */
    export class ExecutionScheduler {
        private runState: 'running' | 'paused' | 'stepping' = 'running';
        private intervalId: NodeJS.Timeout | null = null; // For Node.js timeout
        private browserTimeoutId: number | null = null; // For browser setTimeout
        private isProcessing: boolean = false; // Prevent concurrent processing cycles

        constructor(
            private db: DBInterface,
            private engine: InferenceEngine,
            private handler: ActionHandler,
            private tickInterval: number = 50 // ms between checking for work
        ) {
            console.log("ExecutionScheduler: Initialized.");
            this.scheduleNextTick();
        }

        /** Schedules the next processing tick using appropriate timer. */
        private scheduleNextTick() {
            if (this.runState === 'paused') return; // Don't schedule if paused

            // Clear previous timeouts if any
            if (this.intervalId) clearTimeout(this.intervalId);
            if (this.browserTimeoutId) clearTimeout(this.browserTimeoutId);

            const tickFn = async () => {
                if (this.runState === 'running' || this.runState === 'stepping') {
                    await this.tick();
                    if (this.runState === 'stepping') {
                        this.runState = 'paused'; // Pause after one step
                        console.log("ExecutionScheduler: Paused after step.");
                    }
                }
                // Schedule the next tick unless paused
                if (this.runState !== 'paused') {
                    this.scheduleNextTick();
                }
            };

            if (typeof window !== 'undefined') {
                this.browserTimeoutId = window.setTimeout(tickFn, this.tickInterval);
            } else {
                this.intervalId = setTimeout(tickFn, this.tickInterval);
            }
        }

        /** Performs one processing cycle: find thought, infer, handle actions. */
        private async tick() {
            if (this.isProcessing) return; // Don't run concurrently
            this.isProcessing = true;

            try {
                // 1. Find the highest priority pending thought
                // Query thoughts with status PENDING, pull their goal value.
                const pendingThoughts = this.db.query(`
                     [:find ?uuid ?goalValue
                      :where [?t :kind :flowmind.kind/thought]
                             [?t :thought/status :flowmind.status/pending]
                             [?t :uuid ?uuid]
                             [?t :thought/goal ?g]
                             [?g :goal/value ?goalValue]]
                 `);

                if (pendingThoughts.length === 0) {
                    // console.log("Scheduler.tick: No pending thoughts.");
                    this.isProcessing = false;
                    return; // No work to do
                }

                // 2. Select thought (simple highest goal value for now)
                // Could use probabilistic sampling here too like in engine rule selection.
                let bestThoughtUUID: Types.UUID | null = null;
                let highestGoal = -Infinity;
                for (const [uuid, goalValue] of pendingThoughts) {
                    if (goalValue > highestGoal) {
                        highestGoal = goalValue;
                        bestThoughtUUID = uuid;
                    }
                }

                if (!bestThoughtUUID) {
                    console.error("Scheduler.tick: Failed to select a thought despite pending ones existing.");
                    this.isProcessing = false;
                    return;
                }
                const triggerUUID = bestThoughtUUID;
                console.log(`Scheduler.tick: Selected thought ${triggerUUID.substring(0,8)} (Goal: ${highestGoal.toFixed(2)})`);

                // 3. Attempt to transition thought to ACTIVE
                const thoughtEid = this.db.findEntityIdByAttribute('uuid', triggerUUID);
                if (!thoughtEid) {
                    console.warn(`Scheduler.tick: Selected thought ${triggerUUID} disappeared before activation.`);
                    this.isProcessing = false;
                    return;
                }
                // Use transact to ensure atomicity (though Datascript JS is single-threaded)
                // We check status again *inside* transact implicitly if using compare-and-swap,
                // but a simple update is usually fine here.
                const lockTxReport = this.db.transact([{ ':db/id': thoughtEid, ':thought/status': Status.ACTIVE, modifiedAt: new Date() }]);

                if (!lockTxReport) {
                    console.warn(`Scheduler.tick: Failed to lock thought ${triggerUUID} (maybe already processed?).`);
                    this.isProcessing = false;
                    return; // Failed to acquire lock (e.g., transaction failed)
                }

                // 4. Run inference engine
                const inferenceResult = this.engine.infer(triggerUUID);

                // 5. Process results
                if (inferenceResult) {
                    // Transact DB changes from inference
                    if (inferenceResult.transactionData.length > 0) {
                        this.db.transact(inferenceResult.transactionData);
                    }
                    // Enqueue actions for the handler
                    if (inferenceResult.actions.length > 0) {
                        this.handler.enqueue(inferenceResult.actions);
                    }
                } else {
                    // Inference failed or returned null (e.g., thought wasn't active)
                    // Revert status back to pending or set to failed? Revert for now.
                    console.warn(`Scheduler.tick: Inference returned null for ${triggerUUID}. Reverting status.`);
                    this.db.transact([{ ':db/id': thoughtEid, ':thought/status': Status.PENDING, modifiedAt: new Date() }]);
                }

            } catch (error: any) {
                console.error(`Scheduler.tick: Error during processing cycle: ${error.message}`, error);
                // Consider adding global error event to DB
            } finally {
                this.isProcessing = false;
            }
        }

        /** Sets the run state of the scheduler. */
        setRunState(state: 'running' | 'paused' | 'stepping') {
            if (state === this.runState) return;
            console.log(`ExecutionScheduler: Setting state to ${state}`);
            const previousState = this.runState;
            this.runState = state;

            // If transitioning from paused/stepping to running, kick off the loop
            if (state === 'running' && (previousState === 'paused' || previousState === 'stepping')) {
                this.scheduleNextTick();
            }
            // If setting to stepping, just set state. The current/next tick will handle pausing.
            // If setting to paused, the scheduleNextTick check will prevent rescheduling.
        }

        /** Gets the current run state. */
        getRunState(): 'running' | 'paused' | 'stepping' {
            return this.runState;
        }

        /** Triggers a single processing step if paused. */
        step() {
            if (this.runState === 'paused') {
                console.log("ExecutionScheduler: Stepping...");
                this.runState = 'stepping';
                this.scheduleNextTick(); // Schedule one tick
            } else {
                console.log("ExecutionScheduler: Step ignored, not paused.");
            }
        }

        /** Stops the scheduler permanently. */
        stop() {
            console.log("ExecutionScheduler: Stopping...");
            this.runState = 'paused'; // Set to paused to prevent rescheduling
            if (this.intervalId) clearTimeout(this.intervalId);
            if (this.browserTimeoutId) clearTimeout(this.browserTimeoutId);
            this.intervalId = null;
            this.browserTimeoutId = null;
        }
    }
}


// ======== UI (React) ========
namespace UI {
    import DBInterface = Database.DBInterface;
    import ExecutionScheduler = Scheduler.ExecutionScheduler;
    import Config = Types.Config;
    import Thought = Types.Thought;
    import Status = Types.Status;
    import UIPrompt = Types.UIPrompt;
    import Rule = Types.Rule;
    import Event = Types.Event;

    /** Main React component for the FlowMind UI. */
    export const FlowMindUI: React.FC<{
        db: DBInterface;
        scheduler: ExecutionScheduler;
        initialConfig: Config;
        onSave: () => void;
        onLoad: () => void;
    }> = ({ db, scheduler, initialConfig, onSave, onLoad }) => {
        const [config, setConfig] = useState<Config>(initialConfig);
        const [thoughts, setThoughts] = useState<Thought[]>([]);
        const [rules, setRules] = useState<Rule[]>([]);
        const [events, setEvents] = useState<Event[]>([]);
        const [input, setInput] = useState('');
        const [prompts, setPrompts] = useState<UIPrompt[]>([]);
        const [debugVisible, setDebugVisible] = useState(false);
        const [settingsVisible, setSettingsVisible] = useState(false);
        const [error, setError] = useState<string | null>(null);
        const [runState, setRunState] = useState<'running' | 'paused' | 'stepping'>(scheduler.getRunState());
        const [currentConfigDraft, setCurrentConfigDraft] = useState<Config>(initialConfig);

        // Function to query data from the DB and update UI state
        const updateUIState = useCallback(() => {
            // console.log("UI: Updating state from DB...");
            try {
                // === Thoughts ===
                const thoughtEids = db.query('[:find ?t :where [?t :kind :flowmind.kind/thought]]').map(r => r[0]);
                const thoughtData = db.pullMany(thoughtEids, [
                    'uuid', ':thought/status', ':created_at', ':modified_at',
                    { ':thought/content': ['*'] }, // Pull content details
                    { ':thought/goal': ['*'] },
                    { ':thought/truth': ['*'] },
                    { ':thought/meta': ['*', { ':meta_entry/value_ref': ['uuid'] }] } // Pull meta, including refs if needed
                ]);

                const currentThoughts: Thought[] = thoughtData.map((tData: any) => {
                    if (!tData) return null;
                    const contentTerm = TermUtils.entityToTerm(db, tData[':thought/content']?.[':db/id']);
                    const metaMap = Database.DBInterface.parseMeta(tData[':thought/meta'] ?? []);

                    return {
                        ':db/id': tData[':db/id'],
                        uuid: tData.uuid,
                        content: contentTerm || { kind: 'atom', name: 'Error:DecodingContent' },
                        contentRef: tData[':thought/content']?.[':db/id'],
                        status: tData[':thought/status'],
                        truth: { pos: tData[':thought/truth']?.[':truth/pos'] ?? 0, neg: tData[':thought/truth']?.[':truth/neg'] ?? 0 },
                        goal: {
                            value: tData[':thought/goal']?.[':goal/value'] ?? 0,
                            source: tData[':thought/goal']?.[':goal/source'] ?? 'unknown',
                            time: tData[':thought/goal']?.[':goal/time'] ?? new Date(0)
                        },
                        meta: metaMap,
                        createdAt: tData.createdAt || new Date(0),
                        modifiedAt: tData.modifiedAt || new Date(0),
                    };
                }).filter(Boolean) as Thought[];

                // Sort thoughts (e.g., by creation date descending)
                currentThoughts.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
                setThoughts(currentThoughts);


                // === Prompts ===
                const currentPrompts: UIPrompt[] = currentThoughts
                    .filter(t => t.status === Status.WAITING && t.meta.has(META_KEYS.WAITING_FOR_USER_INPUT))
                    .map(t => ({
                        promptThoughtUUID: t.uuid,
                        // Assume the prompt text is stored in meta or content
                        text: t.meta.get(META_KEYS.UI_PROMPT_TEXT) || TermUtils.formatTerm(t.content),
                        waitingThoughtUUID: t.uuid // The prompt thought itself is waiting
                    }));
                setPrompts(currentPrompts);


                // === Rules (for debug) ===
                if (debugVisible) {
                    const ruleEids = db.query('[:find ?r :where [?r :kind :flowmind.kind/rule]]').map(r => r[0]);
                    const ruleData = db.pullMany(ruleEids, [
                        'uuid', ':created_at', ':modified_at',
                        { ':rule/head': ['*'] },
                        { ':rule/body': ['*'] },
                        { ':rule/truth': ['*'] },
                        { ':rule/meta': ['*', { ':meta_entry/value_ref': ['uuid'] }] }
                    ]);
                    const currentRules: Rule[] = ruleData.map((rData: any) => {
                        if (!rData) return null;
                        const headTerm = TermUtils.entityToTerm(db, rData[':rule/head']?.[':db/id']);
                        const bodyTerms = (rData[':rule/body'] ?? []).map((bRef: any) => TermUtils.entityToTerm(db, bRef[':db/id'])).filter(Boolean);
                        const metaMap = Database.DBInterface.parseMeta(rData[':rule/meta'] ?? []);

                        return {
                            ':db/id': rData[':db/id'],
                            uuid: rData.uuid,
                            head: headTerm || { kind: 'atom', name: 'Error:DecodingHead' },
                            headRef: rData[':rule/head']?.[':db/id'],
                            body: bodyTerms,
                            bodyRefs: (rData[':rule/body'] ?? []).map((bRef: any) => bRef[':db/id']),
                            truth: { pos: rData[':rule/truth']?.[':truth/pos'] ?? 0, neg: rData[':rule/truth']?.[':truth/neg'] ?? 0 },
                            meta: metaMap,
                            createdAt: rData.createdAt || new Date(0),
                            modifiedAt: rData.modifiedAt || new Date(0),
                        };
                    }).filter(Boolean) as Rule[];
                    currentRules.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
                    setRules(currentRules);


                    // === Events (for debug) ===
                    const eventEids = db.query('[:find ?e :where [?e :kind :flowmind.kind/event]]').map(r => r[0]);
                    const eventData = db.pullMany(eventEids, ['uuid', ':event/type', ':event/target_uuid', ':event/data', ':time']);
                    const currentEvents: Event[] = eventData.map((eData: any): Event | null => {
                        if (!eData) return null;
                        return {
                            ':db/id': eData[':db/id'],
                            uuid: eData.uuid,
                            type: eData[':event/type'],
                            targetUUID: eData[':event/target_uuid'],
                            data: eData[':event/data'], // Keep raw data from DB
                            time: eData[':time'] || new Date(0),
                        };
                    }).filter(Boolean) as Event[];
                    currentEvents.sort((a, b) => b.time.getTime() - a.time.getTime()); // Newest first
                    setEvents(currentEvents.slice(0, 50)); // Limit displayed events
                } else {
                    // Clear debug data if not visible to save memory/processing
                    setRules([]);
                    setEvents([]);
                }

                setError(null); // Clear previous errors on successful update
            } catch (e: any) {
                console.error("UI State Update Error:", e);
                setError(`Failed to update UI: ${e.message}`);
            }
        }, [db, debugVisible]); // Dependency: update when db changes or debug visibility changes

        // Effect to subscribe to DB updates
        useEffect(() => {
            console.log("UI: Subscribing to DB updates.");
            db.addListener(updateUIState);
            updateUIState(); // Initial load

            // Cleanup subscription on unmount
            return () => {
                console.log("UI: Unsubscribing from DB updates.");
                db.removeListener(updateUIState);
            };
        }, [db, updateUIState]); // updateUIState is stable due to useCallback

        // Effect to update local run state if scheduler changes it externally
        useEffect(() => {
            // This might need a listener on the scheduler if its state can change externally
            // For now, assume changes are driven by UI buttons
            setRunState(scheduler.getRunState());
        }, [scheduler]);


        const handleAddNote = useCallback(() => {
            const trimmedInput = input.trim();
            if (!trimmedInput) {
                setError('Note content cannot be empty.');
                return;
            }
            try {
                const newThoughtUUID = uuid();
                console.log(`UI: Adding note: "${trimmedInput}" (${newThoughtUUID.substring(0,8)})`);

                // Create the basic 'atom' term for the input text
                const contentTermData = TermUtils.termToTransactionData({ kind: 'atom', name: trimmedInput });
                const goalId = ds.tempid('goal');
                const truthId = ds.tempid('truth');

                db.transact([
                    ...contentTermData.tx,
                    { ':db/id': goalId, ':goal/value': 0.7, ':goal/source': 'user_input', ':goal/time': new Date() },
                    { ':db/id': truthId, ':truth/pos': 1, ':truth/neg': 0 }, // User input is initially trusted
                    {
                        ':db/id': ds.tempid('thought'),
                        uuid: newThoughtUUID,
                        kind: ':flowmind.kind/thought',
                        ':thought/content': contentTermData.rootId,
                        ':thought/status': Status.PENDING, // Start as pending for the engine
                        ':thought/truth': truthId,
                        ':thought/goal': goalId,
                        // No specific meta needed initially unless we add tags, etc.
                        ':thought/meta': [],
                        createdAt: new Date(),
                        modifiedAt: new Date(),
                    }
                ]);
                setInput(''); // Clear input field
                setError(null);
            } catch (e: any) {
                console.error("Add Note Error:", e);
                setError(`Failed to add note: ${e.message}`);
            }
        }, [input, db]);

        const handleRespondToPrompt = useCallback((promptUUID: Types.UUID, response: string) => {
            const trimmedResponse = response.trim();
            if (!trimmedResponse) {
                setError('Response cannot be empty.');
                return;
            }

            // Find the original prompt thought to link the response
            const promptThoughtEid = db.findEntityIdByAttribute('uuid', promptUUID);
            if (!promptThoughtEid) {
                setError(`Prompt thought ${promptUUID} not found.`);
                return;
            }

            try {
                const responseThoughtUUID = uuid();
                console.log(`UI: Responding to prompt ${promptUUID.substring(0,8)} with response thought ${responseThoughtUUID.substring(0,8)}`);

                // Create term for the response
                const responseTermData = TermUtils.termToTransactionData({ kind: 'atom', name: trimmedResponse });
                const goalId = ds.tempid('goal');
                const truthId = ds.tempid('truth');

                // Create meta linking back to the prompt
                const responseMeta = new Map<string, any>([[META_KEYS.RESPONSE_TO_PROMPT, promptUUID]]);
                const metaTxData = Database.DBInterface.buildMetaTx(responseMeta);

                db.transact([
                    // Create the response thought
                    ...responseTermData.tx,
                    ...metaTxData.tx,
                    { ':db/id': goalId, ':goal/value': 0.8, ':goal/source': 'user_response', ':goal/time': new Date() },
                    { ':db/id': truthId, ':truth/pos': 1, ':truth/neg': 0 },
                    {
                        ':db/id': ds.tempid('thought'),
                        uuid: responseThoughtUUID,
                        kind: ':flowmind.kind/thought',
                        ':thought/content': responseTermData.rootId,
                        ':thought/status': Status.PENDING, // Response might trigger rules
                        ':thought/truth': truthId,
                        ':thought/goal': goalId,
                        ':thought/meta': metaTxData.refs,
                        createdAt: new Date(),
                        modifiedAt: new Date(),
                    },
                    // Mark the original prompt thought as DONE (or maybe just remove the WAITING_FOR_USER_INPUT meta?)
                    // Setting to DONE seems reasonable.
                    { ':db/id': promptThoughtEid, ':thought/status': Status.DONE, modifiedAt: new Date() }
                    // We might also need to remove the WAITING_FOR_USER_INPUT meta explicitly if needed
                    // { ':db/id': promptThoughtEid, ':meta/-': [[META_KEYS.WAITING_FOR_USER_INPUT, true]] } // Datascript retract syntax might differ
                ]);
                setError(null);
                // The UI will update automatically via the listener, removing the prompt
            } catch (e: any) {
                console.error("Respond to Prompt Error:", e);
                setError(`Failed to submit response: ${e.message}`);
            }
        }, [db]);


        const handleUpdateGoal = useCallback((thoughtUUID: Types.UUID, delta: number) => {
            const thoughtEid = db.findEntityIdByAttribute('uuid', thoughtUUID);
            if (!thoughtEid) return;

            // Pull the current goal value to update it relatively
            const thoughtGoal = db.pull(thoughtEid, [{ ':thought/goal': [':goal/value'] }])?.[':thought/goal'];
            if (!thoughtGoal) return;

            const currentValue = thoughtGoal[':goal/value'] ?? 0.5;
            const newValue = Math.max(0, Math.min(1, currentValue + delta)); // Clamp between 0 and 1

            const goalEid = db.query('[:find ?g . :where [?t :uuid ?uuid] [?t :thought/goal ?g]]', thoughtUUID)?.[0];

            if (goalEid) {
                db.transact([{
                    ':db/id': goalEid,
                    ':goal/value': newValue,
                    ':goal/source': 'user_adjust',
                    ':goal/time': new Date()
                }]);
            }
        }, [db]);

        const handleToggleSettings = useCallback(() => {
            if (settingsVisible) {
                // Reset draft if closing without saving
                setCurrentConfigDraft(config);
            }
            setSettingsVisible(!settingsVisible);
        }, [settingsVisible, config]);

        const handleSaveSettings = useCallback(() => {
            // TODO: Add validation for config values (e.g., URLs, numbers)
            try {
                setConfig(currentConfigDraft); // Update runtime config
                // TODO: Need to notify relevant components (Tools, Scheduler) about config change
                // This might require passing setConfig down or using a context/event bus.
                // For now, assumes components read config on creation or periodically. Restart might be needed.
                console.log("UI: Settings saved (Note: Restart might be needed for changes to fully apply).", currentConfigDraft);
                setSettingsVisible(false);
                setError(null);
            } catch (e: any) {
                console.error("Save Settings Error:", e);
                setError(`Failed to save settings: ${e.message}`);
            }
        }, [currentConfigDraft]);

        const handleSchedulerControl = useCallback((command: 'run' | 'pause' | 'step') => {
            switch (command) {
                case 'run': scheduler.setRunState('running'); break;
                case 'pause': scheduler.setRunState('paused'); break;
                case 'step': scheduler.step(); break; // Step handles state internally
            }
            // Update UI state representation after a short delay to allow scheduler state change
            setTimeout(() => setRunState(scheduler.getRunState()), 50);
        }, [scheduler]);

        const handleUndo = useCallback(() => {
            console.log("UI: Undoing last transaction.");
            db.undo();
        }, [db]);

        const handleExport = useCallback(() => {
            const exportedData = db.export();
            const blob = new Blob([exportedData], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `flowmind_backup_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            console.log("UI: Database exported.");
        }, [db]);

        const handleImport = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const json = e.target?.result as string;
                    if (db.import(json)) {
                        console.log("UI: Database imported successfully.");
                        setError(null);
                        // Manually trigger UI update after import
                        updateUIState();
                    } else {
                        setError("Failed to import database. Check console for details.");
                    }
                } catch (err: any) {
                    console.error("Import error:", err);
                    setError(`Import failed: ${err.message}`);
                } finally {
                    // Reset file input to allow importing the same file again
                    event.target.value = '';
                }
            };
            reader.onerror = () => {
                setError(`Error reading file: ${reader.error}`);
                event.target.value = '';
            }
            reader.readAsText(file);
        }, [db, updateUIState]);

        const renderTerm = (term: Types.Term): React.ReactNode => {
            // Simple term renderer - could be expanded
            return <span title={`Kind: ${term.kind}`}>{TermUtils.formatTerm(term)}</span>;
        };

        return (
            <div style={styles.container}>
            <div style={styles.header}>
            <h1 style={styles.title}>FlowMind</h1>
        {error && <div style={styles.errorBox}>{error}</div>}
            </div>

            {/* Action Bar */}
            <div style={styles.actionBar}>
            <input
                type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAddNote()}
            placeholder="Add a new thought or task..."
            style={styles.inputField}
            />
            <button onClick={handleAddNote} style={{...styles.button, ...styles.buttonPrimary}}>Add</button>
        <button onClick={handleUndo} style={{...styles.button, ...styles.buttonSecondary}}>Undo</button>
        <button onClick={onSave} style={{...styles.button, ...styles.buttonSuccess}}>Save DB</button>
        <button onClick={onLoad} style={{...styles.button, ...styles.buttonWarning}}>Load DB</button>
        <button onClick={handleToggleSettings} style={{...styles.button, ...styles.buttonInfo}}>{settingsVisible ? 'Hide Settings' : 'Settings'}</button>
        <button onClick={() => setDebugVisible(!debugVisible)} style={{...styles.button, ...styles.buttonDebug}}>{debugVisible ? 'Hide Debug' : 'Debug'}</button>
        </div>

            {/* Settings Panel */}
            {settingsVisible && (
                <div style={styles.panel}>
                <h3 style={styles.panelTitle}>Settings</h3>
                    <div style={styles.settingItem}>
                <label>Ollama Endpoint:</label>
            <input type="text" style={styles.inputField} value={currentConfigDraft.llmEndpoint} onChange={e => setCurrentConfigDraft({ ...currentConfigDraft, llmEndpoint: e.target.value })} />
            </div>
            <div style={styles.settingItem}>
                <label>Ollama Model:</label>
            <input type="text" style={styles.inputField} value={currentConfigDraft.ollamaModel} onChange={e => setCurrentConfigDraft({ ...currentConfigDraft, ollamaModel: e.target.value })} />
            </div>
            <div style={styles.settingItem}>
                <label>AutoSave Interval (ms):</label>
            <input type="number" style={styles.inputField} value={currentConfigDraft.autoSaveInterval} onChange={e => setCurrentConfigDraft({ ...currentConfigDraft, autoSaveInterval: parseInt(e.target.value) || 30000 })} />
            </div>
                {/* Add other settings here */}
                <button onClick={handleSaveSettings} style={{...styles.button, ...styles.buttonPrimary}}>Save Settings</button>
            <button onClick={handleToggleSettings} style={{...styles.button, ...styles.buttonSecondary, marginLeft: '10px'}}>Cancel</button>
            </div>
            )}

            {/* Prompts Panel */}
            {prompts.length > 0 && (
                <div style={styles.panel}>
                <h3 style={styles.panelTitle}>Waiting for Input</h3>
            {prompts.map(p => (
                <div key={p.promptThoughtUUID} style={styles.promptItem}>
            <p style={{ margin: '0 0 8px 0' }}>{p.text}</p>
            <input
                type="text"
                placeholder="Your response..."
                onKeyDown={e => { if (e.key === 'Enter') handleRespondToPrompt(p.promptThoughtUUID, (e.target as HTMLInputElement).value); }}
                style={{...styles.inputField, marginBottom: '5px'}}
                />
                {/* Add submit button if needed */}
                <button onClick={(e) => {
                const inputElement = (e.target as HTMLElement).previousElementSibling as HTMLInputElement;
                handleRespondToPrompt(p.promptThoughtUUID, inputElement?.value || '');
            }} style={{...styles.button, ...styles.buttonSmall}}>Submit</button>
            </div>
            ))}
                </div>
            )}

            {/* Thoughts List */}
            <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Thoughts</h2>
            {thoughts.length === 0 ? (
                <p style={{ color: '#666' }}>No thoughts yet. Add one above!</p>
            ) : (
                <ul style={styles.thoughtList}>
                    {thoughts.map(t => (
                            <li key={t.uuid} style={styles.thoughtItem}>
                        <div style={styles.thoughtContent}>
                        <span style={styles.thoughtStatus(t.status)}>{t.status.split('/')[1]}</span>
                            <strong style={{ marginLeft: '10px' }}>{renderTerm(t.content)}</strong>
            <div style={styles.thoughtMeta}>
                <span>Priority: {t.goal.value.toFixed(2)} ({t.goal.source})</span>
            <span>Created: {t.createdAt.toLocaleString()}</span>
                {/* Display some meta? */}
                {/* {Array.from(t.meta.entries()).map(([k,v]) => <span key={k}>{k}={v}</span>)} */}
                </div>
                </div>
                <div style={styles.thoughtActions}>
            <button onClick={() => handleUpdateGoal(t.uuid, 0.1)} style={{...styles.button, ...styles.buttonSmall, ...styles.buttonSuccess}} title="Increase Priority">+</button>
                <button onClick={() => handleUpdateGoal(t.uuid, -0.1)} style={{...styles.button, ...styles.buttonSmall, ...styles.buttonDanger}} title="Decrease Priority">-</button>
                {/* Add delete/edit buttons? */}
                </div>
                </li>
            ))}
                </ul>
            )}
            </div>

            {/* Debug Panel */}
            {debugVisible && (
                <div style={styles.panel}>
                <h3 style={styles.panelTitle}>Debug Console</h3>
                {/* Scheduler Controls */}
                <div style={{ marginBottom: '15px', paddingBottom: '15px', borderBottom: '1px solid #eee' }}>
                <span style={{ marginRight: '10px' }}>Scheduler: <strong>{runState}</strong></span>
            <button onClick={() => handleSchedulerControl('pause')} disabled={runState === 'paused'} style={{...styles.button, ...styles.buttonSmall, ...styles.buttonDanger}}>Pause</button>
            <button onClick={() => handleSchedulerControl('step')} disabled={runState !== 'paused'} style={{...styles.button, ...styles.buttonSmall, ...styles.buttonWarning}}>Step</button>
            <button onClick={() => handleSchedulerControl('run')} disabled={runState === 'running'} style={{...styles.button, ...styles.buttonSmall, ...styles.buttonSuccess}}>Run</button>
            </div>

                {/* DB Import/Export */}
                <div style={{ marginBottom: '15px', paddingBottom: '15px', borderBottom: '1px solid #eee' }}>
                <button onClick={handleExport} style={{...styles.button, ...styles.buttonSmall, ...styles.buttonInfo}}>Export DB</button>
            <label style={{...styles.button, ...styles.buttonSmall, ...styles.buttonInfo, marginLeft: '10px', cursor: 'pointer'}}>
                Import DB
            <input type="file" accept=".json" onChange={handleImport} style={{ display: 'none' }} />
            </label>
            </div>


                {/* Rules Display */}
                <div style={styles.debugSection}>
                    <h4>Rules ({rules.length})</h4>
                    <pre style={styles.debugPre}>
                {rules.map(rule => (
                        <div key={rule.uuid} style={{ marginBottom: '5px', paddingBottom: '5px', borderBottom: '1px dotted #ccc' }}>
                <strong>{rule.uuid.substring(0, 8)}:</strong> {TermUtils.formatTerm(rule.head)}<br />
            &nbsp;&nbsp; :- {rule.body.map(b => TermUtils.formatTerm(b)).join(', ')}<br />
            &nbsp;&nbsp; (Truth: {rule.truth.pos}/{rule.truth.neg})
            </div>
            ))}
                </pre>
                </div>

                {/* Events Display */}
                <div style={styles.debugSection}>
                    <h4>Recent Events ({events.length})</h4>
            <pre style={styles.debugPre}>
                {events.map(event => (
                        <div key={event.uuid}>
                            [{new Date(event.time).toLocaleTimeString()}] {event.type}
                {event.targetUUID ? ` (${event.targetUUID.substring(0, 8)})` : ''}: {typeof event.data === 'string' ? event.data.substring(0, 100) : JSON.stringify(event.data)}
                </div>
            ))}
                </pre>
                </div>

                {/* Raw Thoughts (optional) */}
                {/* <div style={styles.debugSection}>
                             <h4>Raw Thoughts Data</h4>
                             <pre style={styles.debugPre}>{JSON.stringify(thoughts, null, 2)}</pre>
                         </div> */}
                </div>
            )}
            </div>
        );
        };


        // Basic Styling
        const styles = {
            container: { fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", padding: '20px', maxWidth: '1000px', margin: '20px auto', background: '#f8f9fa', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' },
            header: { marginBottom: '20px', paddingBottom: '10px', borderBottom: '1px solid #dee2e6' },
            title: { color: '#343a40', margin: '0 0 10px 0' },
            actionBar: { display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' } as React.CSSProperties,
            inputField: { flexGrow: 1, padding: '10px', border: '1px solid #ced4da', borderRadius: '4px', fontSize: '1rem' },
            button: { padding: '10px 15px', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '1rem', transition: 'background-color 0.2s ease' },
            buttonPrimary: { background: '#007bff', color: 'white' },
            buttonSecondary: { background: '#6c757d', color: 'white' },
            buttonSuccess: { background: '#28a745', color: 'white' },
            buttonDanger: { background: '#dc3545', color: 'white' },
            buttonWarning: { background: '#ffc107', color: 'black' },
            buttonInfo: { background: '#17a2b8', color: 'white' },
            buttonDebug: { background: '#fd7e14', color: 'white' },
            buttonSmall: { padding: '5px 10px', fontSize: '0.85rem', marginLeft: '5px' },
            panel: { marginBottom: '20px', padding: '15px', background: 'white', border: '1px solid #dee2e6', borderRadius: '4px' },
            panelTitle: { marginTop: '0', color: '#495057', borderBottom: '1px solid #eee', paddingBottom: '5px', marginBottom: '10px' },
            errorBox: { color: '#721c24', background: '#f8d7da', border: '1px solid #f5c6cb', padding: '10px', borderRadius: '4px', marginTop: '10px' },
            settingItem: { marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' },
            promptItem: { marginBottom: '10px', padding: '10px', background: '#e9ecef', borderRadius: '4px', borderLeft: '3px solid #007bff'},
            thoughtList: { listStyle: 'none', padding: 0, margin: 0 },
            thoughtItem: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 0', borderBottom: '1px solid #eee' },
            thoughtContent: { flexGrow: 1, marginRight: '10px' },
            thoughtActions: { flexShrink: 0 },
            thoughtStatus: (status: Types.StatusValue) => ({
                display: 'inline-block',
                padding: '2px 6px',
                fontSize: '0.8em',
                borderRadius: '4px',
                color: 'white',
                background: status === Status.DONE ? '#28a745' : status === Status.PENDING ? '#ffc107' : status === Status.ACTIVE ? '#007bff' : status === Status.WAITING ? '#17a2b8' : status === Status.FAILED ? '#dc3545' : '#6c757d',
                color: status === Status.PENDING ? 'black' : 'white',
            }),
            thoughtMeta: { fontSize: '0.85em', color: '#6c757d', marginTop: '5px', display: 'flex', gap: '10px', flexWrap: 'wrap'} as React.CSSProperties,
            debugSection: { marginBottom: '15px' },
            debugPre: { background: '#e9ecef', padding: '10px', borderRadius: '4px', maxHeight: '300px', overflowY: 'auto' as 'auto', fontSize: '0.9em', whiteSpace: 'pre-wrap', wordBreak: 'break-all' } as React.CSSProperties,
        };
    }

// ======== Bootstrap Rules ========
    namespace Bootstrap {
        import Term = Types.Term;
        import Rule = Types.Rule;
        import DBInterface = Database.DBInterface;

        /** Defines the initial set of rules for the system. */
        export const getBootstrapRules = (): Omit<Rule, 'createdAt' | 'modifiedAt'>[] => [
            // Rule 1: If a thought is just added (user_input source), embed it and add to memory.
            {
                uuid: uuid(),
                head: { kind: 'struct', name: 'thought', args: [
                        { kind: 'var', name: 'T_UUID' },
                        { kind: 'var', name: 'ContentTerm' }, // Matches any content term
                        { kind: 'atom', name: ':flowmind.status/pending' },
                        { kind: 'struct', name: 'goal', args: [{ kind: 'var', name: '_GVal'}, { kind: 'atom', name: 'user_input' }] } // Matches thoughts from user
                    ]},
                body: [
                    // Log the content
                    { kind: 'struct', name: 'op:log', args: [{ kind: 'struct', name: 'log_msg', args: [{kind: 'atom', name: 'Embedding thought '}, {kind: 'var', name: 'T_UUID'}] }]},
                    // Embed the content (assuming content is atom for simplicity here)
                    { kind: 'struct', name: 'op:tool', args: [
                            { kind: 'atom', name: 'llm' },
                            { kind: 'struct', name: 'embed_text', args: [{ kind: 'var', name: 'ContentTerm' }] },
                            { kind: 'var', name: 'EmbeddingVector' } // Result bound to ?EmbeddingVector
                        ]},
                    // Add to memory store (assuming content is atom - needs robust handling for complex terms)
                    { kind: 'struct', name: 'op:tool', args: [
                            { kind: 'atom', name: 'memory' },
                            { kind: 'struct', name: 'add_entry', args: [
                                    { kind: 'var', name: 'T_UUID' }, // Use thought UUID as ID
                                    { kind: 'var', name: 'ContentTerm' }, // Use original content text
                                    { kind: 'struct', name: 'meta', args: [ { kind: 'struct', name: 'kv', args: [{kind: 'atom', name: 'source'}, {kind: 'atom', name: 'user_input'}] } ]} // Example meta
                                ]},
                            // No result variable needed for add_entry
                        ]},
                    // Set status to done (or maybe pending again if further processing needed?)
                    // Let's make it Done for now, embedding is the final step for this rule.
                    { kind: 'struct', name: 'op:set', args: [{ kind: 'var', name: 'T_UUID' }, { kind: 'atom', name: ':thought/status' }, { kind: 'atom', name: Types.Status.DONE }] },
                ],
                truth: { pos: 5, neg: 0 }, // High confidence in this rule
                meta: new Map([['description', 'Embed user input thoughts and add to memory']]),
            },

            // Rule 2: If no rule matches, try to synthesize one.
            {
                uuid: uuid(),
                head: { kind: 'struct', name: 'event', args: [
                        { kind: 'var', name: 'E_UUID' },
                        { kind: 'atom', name: ':flowmind.event.type/no_rule_match' },
                        { kind: 'var', name: 'T_UUID' },
                        { kind: 'var', name: 'FailedContentTerm' } // The content term that failed
                    ]},
                body: [
                    { kind: 'struct', name: 'op:log', args: [{kind: 'struct', name: 'log_msg', args: [{kind: 'atom', name: 'No rule match for '}, { kind: 'var', name: 'FailedContentTerm' }, {kind: 'atom', name: '. Attempting synthesis.'}]}]},
                    // Mark the original thought as failed permanently
                    { kind: 'struct', name: 'op:set', args: [{ kind: 'var', name: 'T_UUID' }, { kind: 'atom', name: ':thought/status' }, { kind: 'atom', name: Types.Status.FAILED }] },
                    // Ask LLM to synthesize a rule
                    // Note: This currently just gets text back. Needs integration step.
                    { kind: 'struct', name: 'op:tool', args: [
                            { kind: 'atom', name: 'llm' },
                            { kind: 'struct', name: 'synthesize_rule', args: [ { kind: 'struct', name: 'rule_description', args: [{kind: 'atom', name: 'Rule needed for thought: '}, { kind: 'var', name: 'FailedContentTerm' }] }] },
                            { kind: 'var', name: 'SynthesizedRuleText' }
                        ]},
                    // Add a new thought containing the synthesized rule text for user review/action
                    { kind: 'struct', name: 'op:add_thought', args: [
                            { kind: 'struct', name: 'synthesized_rule', args: [{ kind: 'var', name: 'SynthesizedRuleText' }] }, // Content is the rule text
                            { kind: 'struct', name: 'goal', args: [{ kind: 'atom', name: '0.9' }, { kind: 'atom', name: 'rule_synthesis' }] }, // High priority
                            { kind: 'struct', name: 'meta', args: [ { kind: 'struct', name: 'kv', args: [{kind: 'atom', name: 'requires_action'}, {kind: 'atom', name: 'review_synthesized_rule'}]} ]}
                        ]},
                    // Optionally, suggest goals based on the failed content
                    { kind: 'struct', name: 'op:tool', args: [
                            { kind: 'atom', name: 'goal_proposal' },
                            { kind: 'struct', name: 'suggest_goals', args: [{ kind: 'var', name: 'FailedContentTerm' }] },
                            { kind: 'var', name: 'SuggestedGoalList' }
                        ]},
                    // Add a thought for the suggested goals list
                    { kind: 'struct', name: 'op:add_thought', args: [
                            { kind: 'struct', name: 'suggested_goals', args: [{ kind: 'var', name: 'SuggestedGoalList' }] },
                            { kind: 'struct', name: 'goal', args: [{ kind: 'atom', name: '0.7' }, { kind: 'atom', name: 'goal_suggestion' }] },
                            { kind: 'struct', name: 'meta', args: [ { kind: 'struct', name: 'kv', args: [{kind: 'atom', name: 'parent_failed_thought'}, { kind: 'var', name: 'T_UUID'}]} ]}
                        ]},
                ],
                truth: { pos: 1, neg: 0 },
                meta: new Map([['description', 'Handle no_rule_match events by synthesizing a rule and suggesting goals']]),
            },

            // Rule 3: Generic rule - If a thought is active, ask the user for clarification.
            // This is a low-priority fallback rule.
            /* // Commenting out - This can lead to excessive prompting if not constrained.
               // Needs better conditions (e.g., only if confidence is low, or content is ambiguous).
            {
                uuid: uuid(),
                head: { kind: 'struct', name: 'thought', args: [
                    { kind: 'var', name: 'T_UUID' },
                    { kind: 'var', name: 'Content' },
                    { kind: 'atom', name: ':flowmind.status/active' } // Matches any active thought not handled by other rules
                ]},
                body: [
                     { kind: 'struct', name: 'op:log', args: [{kind: 'struct', name: 'log_msg', args: [{kind: 'atom', name: 'Fallback: Prompting user for clarification on '}, { kind: 'var', name: 'Content' }]}]},
                     { kind: 'struct', name: 'op:tool', args: [
                         { kind: 'atom', name: 'user_interaction' },
                         // Create a specific prompt text
                         { kind: 'struct', name: 'prompt', args: [{kind: 'atom', name: 'Need clarification or next step for: '}, {kind: 'var', name: 'Content'}]},
                         // No result variable needed here, just pausing the thought.
                     ]},
                     // Set the current thought to waiting (the tool interaction already does this by creating a prompt thought,
                     // but setting the original thought to waiting ensures it doesn't get picked up again immediately).
                     // Let's let the user_interaction tool handle the waiting state via the prompt thought.
                     // Set original thought to DONE as the prompt handles the next step.
                     { kind: 'struct', name: 'op:set', args: [{ kind: 'var', name: 'T_UUID' }, { kind: 'atom', name: ':thought/status' }, { kind: 'atom', name: Types.Status.DONE }] },
                ],
                truth: { pos: 0, neg: 5 }, // Low confidence, high negative feedback initially
                meta: new Map([['description', 'Fallback rule: Ask user for clarification on active thoughts']]),
            },
            */

            // Rule 4: Example Search Rule - If a thought contains "search for X", use memory search.
            {
                uuid: uuid(),
                head: { kind: 'struct', name: 'thought', args: [
                        { kind: 'var', name: 'T_UUID' },
                        { kind: 'struct', name: 'atom', args: [{ kind: 'var', name: 'SearchQueryAtom'}] }, // Match atom content
                        { kind: 'atom', name: ':flowmind.status/pending' }
                    ]},
                // Condition in body (pseudo): Check if SearchQueryAtom contains "search for"
                // Datascript rules are typically more declarative. We simulate this check
                // by making the rule specific, or by adding metadata indicating a search task.
                // Let's assume the head matches `atom("search for X")` for simplicity.
                // A more robust way involves more complex term matching or LLM intent recognition.
                // For this example, let's refine the head to be more specific:
                head: { kind: 'struct', name: 'thought', args: [
                        { kind: 'var', name: 'T_UUID' },
                        // Match content like: request_search("query text")
                        { kind: 'struct', name: 'request_search', args: [{kind: 'var', name: 'QueryText'}] },
                        { kind: 'atom', name: ':flowmind.status/pending' }
                    ]},
                body: [
                    { kind: 'struct', name: 'op:log', args: [{kind: 'struct', name: 'log_msg', args: [{kind: 'atom', name: 'Performing memory search for: '}, { kind: 'var', name: 'QueryText' }]}]},
                    { kind: 'struct', name: 'op:tool', args: [
                            { kind: 'atom', name: 'memory' },
                            { kind: 'struct', name: 'search_similar', args: [
                                    { kind: 'var', name: 'QueryText' },
                                    { kind: 'atom', name: '3' } // Get top 3 results
                                ]},
                            { kind: 'var', name: 'SearchResultsList' } // Bind results
                        ]},
                    // Add a new thought containing the search results
                    { kind: 'struct', name: 'op:add_thought', args: [
                            { kind: 'struct', name: 'search_results', args: [{ kind: 'var', name: 'SearchResultsList' }] },
                            { kind: 'struct', name: 'goal', args: [{ kind: 'atom', name: '0.8' }, { kind: 'atom', name: 'search_result_display' }] },
                            { kind: 'struct', name: 'meta', args: [ { kind: 'struct', name: 'kv', args: [{kind: 'atom', name: 'parent_search_request'}, { kind: 'var', name: 'T_UUID'}]} ]}
                        ]},
                    // Mark original search request thought as DONE
                    { kind: 'struct', name: 'op:set', args: [{ kind: 'var', name: 'T_UUID' }, { kind: 'atom', name: ':thought/status' }, { kind: 'atom', name: Types.Status.DONE }] },
                ],
                truth: { pos: 2, neg: 0 },
                meta: new Map([['description', 'Perform memory search for thoughts like request_search("...")']]),
            },
        ];

        /** Adds the bootstrap rules to the database. */
        export const bootstrapDB = (db: DBInterface) => {
            console.log("Bootstrapping database with initial rules...");
            const rules = getBootstrapRules();
            let transactionData: any[] = [];

            for (const rule of rules) {
                const headTxData = TermUtils.termToTransactionData(rule.head);
                transactionData.push(...headTxData.tx);
                const headRef = headTxData.rootId;

                const bodyTxData = rule.body.map(op => TermUtils.termToTransactionData(op));
                transactionData.push(...bodyTxData.flatMap(d => d.tx));
                const bodyRefs = bodyTxData.map(d => d.rootId);

                const truthId = ds.tempid('truth');
                transactionData.push({ ':db/id': truthId, ':truth/pos': rule.truth.pos, ':truth/neg': rule.truth.neg });

                const metaTxData = Database.DBInterface.buildMetaTx(rule.meta);
                transactionData.push(...metaTxData.tx);

                transactionData.push({
                    ':db/id': ds.tempid('rule'),
                    uuid: rule.uuid,
                    kind: ':flowmind.kind/rule',
                    ':rule/head': headRef,
                    ':rule/body': bodyRefs,
                    ':rule/truth': truthId,
                    ':rule/meta': metaTxData.refs,
                    createdAt: new Date(), // Set creation time during bootstrap
                    modifiedAt: new Date(),
                });
            }

            // Add a welcome thought
            const welcomeUUID = uuid();
            const welcomeTerm = TermUtils.termToTransactionData({kind: 'atom', name: 'Welcome to FlowMind! Add a note or task above to get started.'});
            transactionData.push(...welcomeTerm.tx);
            transactionData.push({
                ':db/id': ds.tempid('thought'),
                uuid: welcomeUUID,
                kind: ':flowmind.kind/thought',
                ':thought/content': welcomeTerm.rootId,
                ':thought/status': Types.Status.DONE, // Welcome message is just informational
                ':thought/truth': {':db/id': ds.tempid('truth'), ':truth/pos': 1, ':truth/neg': 0},
                ':thought/goal': {':db/id': ds.tempid('goal'), ':goal/value': 0.1, ':goal/source': 'system', ':goal/time': new Date()},
                createdAt: new Date(),
                modifiedAt: new Date(),
            });

            db.transact(transactionData);
            console.log(`Bootstrap complete. Added ${rules.length} rules and a welcome thought.`);
        };
    }


// ======== Main Application Setup ========
    const main = () => {
        console.log("FlowMind: Initializing application...");

        // --- Configuration ---
        // In a real app, load this from storage or environment
        const config = { ...DEFAULT_CONFIG };
        console.log("FlowMind: Using configuration:", config);

        // --- Database ---
        const db = new Database.DBInterface();

        // --- Persistence ---
        const loadDatabase = () => {
            if (typeof window !== 'undefined' && window.localStorage) {
                const savedDB = localStorage.getItem(DB_STORAGE_KEY);
                if (savedDB) {
                    console.log("FlowMind: Loading database from LocalStorage...");
                    if (db.import(savedDB)) {
                        console.log("FlowMind: Database loaded successfully.");
                    } else {
                        console.error("FlowMind: Failed to load database from LocalStorage, starting fresh.");
                        // Optionally clear corrupted data: localStorage.removeItem(DB_STORAGE_KEY);
                        Bootstrap.bootstrapDB(db); // Load default rules if load failed
                    }
                } else {
                    console.log("FlowMind: No saved database found, bootstrapping initial state.");
                    Bootstrap.bootstrapDB(db); // Load default rules if no saved state
                }
            } else {
                console.warn("FlowMind: LocalStorage not available, database will not persist.");
                Bootstrap.bootstrapDB(db); // Load default rules
            }
        };

        const saveDatabase = () => {
            if (typeof window !== 'undefined' && window.localStorage) {
                try {
                    console.log("FlowMind: Saving database to LocalStorage...");
                    const exportedData = db.export();
                    localStorage.setItem(DB_STORAGE_KEY, exportedData);
                    console.log("FlowMind: Database saved.");
                } catch (e: any) {
                    console.error(`FlowMind: Error saving database: ${e.message}`);
                    // Handle potential storage quota errors
                    if (e.name === 'QuotaExceededError') {
                        alert("Error: LocalStorage quota exceeded. Cannot save database. Please export your data manually if needed.");
                    }
                }
            }
        };

        // Initial load
        loadDatabase();

        // Auto-save interval
        let autoSaveTimer: NodeJS.Timeout | null = null;
        if (typeof window !== 'undefined') {
            autoSaveTimer = setInterval(saveDatabase, config.autoSaveInterval);
        }


        // --- Core Components ---
        const tools = new Tools.ToolRegistry(config, db);
        const engine = new Engine.InferenceEngine(db);
        const handler = new ActionHandler.ActionHandler(db, tools);
        const scheduler = new Scheduler.ExecutionScheduler(db, engine, handler);


        // --- UI Rendering ---
        const rootElement = document.getElementById('root');
        if (!rootElement) {
            console.error("FlowMind: Root element 'root' not found in the DOM.");
            return;
        }
        const reactRoot = createRoot(rootElement);
        reactRoot.render(
            <React.StrictMode>
                <UI.FlowMindUI
                    db={db}
        scheduler={scheduler}
        initialConfig={config}
        onSave={saveDatabase}
        onLoad={loadDatabase}
        />
        </React.StrictMode>
    );

        // --- Cleanup ---
        // Ensure scheduler stops and saves on page unload (best effort)
        if (typeof window !== 'undefined') {
            window.addEventListener('beforeunload', () => {
                console.log("FlowMind: Unloading application...");
                scheduler.stop();
                if(autoSaveTimer) clearInterval(autoSaveTimer);
                saveDatabase(); // Attempt a final save
            });
        }

        console.log("FlowMind: Application initialized and UI rendered.");
    };

// --- Start the application ---
// Ensure DOM is ready for React rendering in browser environments
    if (typeof window !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', main);
        } else {
            main(); // DOM already ready
        }
    } else {
        // Running in Node.js (e.g., for testing or server-side logic)
        // The UI part won't render, but core logic can be used.
        console.log("FlowMind: Running in non-browser environment. UI skipped.");
        // main(); // You might call main() differently in Node.js context if needed
    }
//
//     ```
//
// **Key Changes and Improvements:**
//
// 1.  **LangChain Updates:**
//     *   Uses `ChatOllama` and `OllamaEmbeddings` from `@langchain/community`.
//     *   Uses `MemoryVectorStore` from `@langchain/community` which requires `Embeddings` during initialization.
//     *   `MemoryTool` now handles embedding internally using the provided `OllamaEmbeddings`. It takes text content as input.
//     *   `LLMTool` uses `OllamaEmbeddings` for the `embed_text` operation.
//     *   Imports adjusted for latest package structure (`@langchain/core`, `@langchain/community`).
//
// 2.  **Structure & Readability:**
//     *   Code is organized into namespaces (`Types`, `Config`, `TermUtils`, `Database`, `Tools`, `Engine`, `ActionHandler`, `Scheduler`, `UI`, `Bootstrap`).
//     *   Added JSDoc comments to explain major components and types.
//     *   Used an `enum` for `Status` for better type safety and readability.
//     *   Standardized meta keys using `META_KEYS` constant.
//
// 3.  **Database (`DBInterface`):**
//     *   Added `addListener`, `removeListener`, `notifyListeners` for reactive UI updates.
//     *   `transact` now calls `notifyListeners`.
//     *   Added basic `load`/`save` methods using `localStorage` via `export`/`import`.
//     *   Improved schema with `:db/doc` strings.
//     *   Added static helpers `buildMetaTx` and `parseMeta` for handling metadata entities more consistently.
//     *   Added `findEntityIdByAttribute` helper.
//     *   Improved `pull` and added `pullMany`.
//     *   Timestamps (`createdAt`, `modifiedAt`) are automatically handled in `transact`.
//
// 4.  **Term Handling (`TermUtils`):**
//     *   Renamed `toTerm` to `entityToTerm` and `fromTerm` to `termToTransactionData` for clarity.
//     *   Improved robustness slightly, but the core complexity remains due to the translation layer.
//     *   Added `formatTerm` for better display/logging.
//     *   Added `substitute` function to apply bindings to terms before use in operations.
//     *   Refined `unify` logic slightly.
//
// 5.  **Tools:**
//     *   `LLMTool` & `MemoryTool` updated as per LangChain changes. `MemoryTool` now embeds text itself.
//     *   `UserInteractionTool`: Removed direct UI callback. Now creates a dedicated "prompt" thought with status `WAITING` and metadata (`META_KEYS.UI_PROMPT_TEXT`, `META_KEYS.WAITING_FOR_USER_INPUT`). The UI queries for these thoughts.
//     *   `CollaborationTool`: Commented out as it was a placeholder and requires external setup.
//     *   `GoalProposalTool`: Uses `LLMTool` for generation; includes basic parsing (needs improvement).
//     *   `ToolRegistry`: Updated instantiation to pass necessary dependencies (config, db).
//
// 6.  **Engine (`InferenceEngine`):**
//     *   Uses more specific pull patterns.
//     *   Applies `TermUtils.substitute` to operation terms before execution.
//     *   Improved handling of `op:tool` results: If a `resultBindingVar` is present, the `ActionHandler` is now expected to create a *new* thought with the result and potentially reactivate the waiting thought. The original rule sets the trigger thought to `WAITING`.
//     *   Added `op:log` operation.
//     *   Handles `op:set` for basic atoms and attempts reference setting for complex terms.
//     *   Creates DB events for significant occurrences (rule execution, failures, etc.).
//     *   Refined rule selection logic.
//
// 7.  **Action Handler (`ActionHandler`):**
//     *   Processes actions sequentially.
//     *   Improved logic for handling tool results: When a tool returns a result for a `resultBindingVar`, it now creates a *new* thought containing the result (linked via `META_KEYS.TOOL_RESULT_FOR`) and sets the original waiting thought back to `PENDING`.
//     *   Logs events for tool success/failure.
//     *   Handles tool errors by potentially marking the waiting thought as `FAILED`.
//     *   Basic handling for `log` actions.
//
// 8.  **Scheduler (`ExecutionScheduler`):**
//     *   Uses `setTimeout` instead of `setInterval` for better control over tick scheduling.
//     *   Selects thoughts based on highest goal value (could be enhanced with sampling).
//     *   Handles locking/unlocking (`ACTIVE` status) more explicitly.
//     *   Includes basic state management (`running`, `paused`, `stepping`).
//
// 9.  **UI (`FlowMindUI`):**
//     *   **Reactivity:** Removed `setInterval`. Uses `useEffect` to subscribe to `db.addListener` and updates state via `updateUIState` callback.
//     *   **Prompt Handling:** Queries for thoughts with `status === WAITING` and `META_KEYS.WAITING_FOR_USER_INPUT` metadata to display prompts. `handleRespondToPrompt` now finds the prompt thought, creates a response thought, and marks the prompt thought as `DONE`.
//     *   **State Updates:** `updateUIState` pulls data using more specific patterns and reconstructs the state objects (`Thought`, `Rule`, `Event`, `UIPrompt`).
//     *   **Persistence:** Added Save/Load DB buttons triggering `onSave`/`onLoad` props (connected to `db.export`/`import`). Added file import button.
//     *   **Debug View:** Added display for Rules and recent Events. Includes DB Export/Import buttons.
//     *   **Styling:** Basic CSS-in-JS styling for a cleaner look.
//     *   **Settings:** Allows editing config (requires restart/reload for tools/scheduler to pick up changes currently).
//     *   **Goal Updates:** Buttons now call `handleUpdateGoal` which correctly finds the goal entity and updates its value.
//     *   **Add Note:** Creates a `PENDING` thought for the engine to process (e.g., embedding via Rule 1).
//
// 10. **Bootstrap Rules:**
//     *   Rewritten to use the new `op:` syntax and structures.
//     *   **Rule 1:** Added rule to automatically embed `user_input` thoughts and add them to memory.
//     *   **Rule 2:** Handles `no_rule_match` events, attempts rule synthesis (returns text for now), suggests goals, and marks original thought as failed.
//     *   **Rule 3 (Fallback Prompt):** Commented out as it can be overly aggressive.
//     *   **Rule 4:** Example rule demonstrating memory search triggered by a specific thought structure (`request_search(...)`).
//     *   Bootstrap process now correctly converts terms to transaction data. Added a Welcome thought.
//
// 11. **Main:**
//     *   Initializes components in the correct order.
//     *   Sets up DB loading from LocalStorage on start.
//     *   Sets up DB auto-saving to LocalStorage.
//     *   Renders the React UI using `createRoot`.
//     *   Adds basic `beforeunload` handler for cleanup/final save.
//
// **To Run This:**
//
// 1.  **Dependencies:** Make sure you have Node.js and npm installed. Run:
//     ```bash
//     npm install react react-dom socket.io-client datascript langchain @langchain/community @langchain/core uuid
//     npm install --save-dev @types/react @types/react-dom @types/uuid typescript ts-loader webpack webpack-cli html-webpack-plugin concurrently
//     ```
// 2.  **Ollama:** Ensure Ollama is running locally (usually at `http://localhost:11434`) and has a model pulled (e.g., `ollama pull llama3`). Adjust `ollamaModel` in `DEFAULT_CONFIG` if needed.
//     3.  **Build:** This single file is now quite large and uses dependencies that benefit from bundling. You'll need a build step (like Webpack, Parcel, or Vite) to create a runnable JavaScript file for the browser.
// *   **Example Webpack Setup (Minimal):**
// *   Create `webpack.config.js`:
//     ```javascript
//             const path = require('path');
//             const HtmlWebpackPlugin = require('html-webpack-plugin');
//
//             module.exports = {
//                 entry: './flowmind.ts', // Your filename
//                 mode: 'development', // Or 'production'
//                 module: {
//                     rules: [ { test: /\.tsx?$/, use: 'ts-loader', exclude: /node_modules/ } ]
//                 },
//                 resolve: { extensions: ['.tsx', '.ts', '.js'] },
//                 output: { filename: 'bundle.js', path: path.resolve(__dirname, 'dist') },
//                 plugins: [ new HtmlWebpackPlugin({ templateContent: '<!DOCTYPE html><html><body><div id="root"></div></body></html>' }) ],
//                 devServer: { static: path.join(__dirname, 'dist'), compress: true, port: 9000 }
//             };
//             ```
//     *   Create `tsconfig.json`:
//     ```json
//             {
//               "compilerOptions": {
//                 "outDir": "./dist/",
//                 "noImplicitAny": true,
//                 "module": "ESNext", // Changed for modern compatibility
//                 "target": "es6", // Target modern browsers
//                 "jsx": "react-jsx", // Use new JSX transform
//                 "allowJs": true,
//                 "moduleResolution": "node",
//                 "esModuleInterop": true, // Important for compatibility
//                 "skipLibCheck": true, // Optional: Speeds up build
//                 "forceConsistentCasingInFileNames": true,
//                 "strict": true // Recommended for new code
//               },
//               "include": ["./**/*.ts", "./**/*.tsx"], // Ensure your file is included
//               "exclude": ["node_modules"]
//             }
//             ```
//     *   Save the revised code above as `flowmind.ts`.
// *   Add scripts to `package.json`:
//     ```json
//              "scripts": {
//                "build": "webpack",
//                "start": "webpack serve --open"
//              },
//             ```
//     *   Run `npm start`. This should build the app and open it in your browser.
//
//     This revised version provides a more robust foundation, incorporates modern LangChain features, and adds key functionalities like persistence and improved UI reactivity. Remember that the rule synthesis and collaboration features require further development for full functionality.