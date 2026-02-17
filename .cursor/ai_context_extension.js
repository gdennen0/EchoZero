/**
 * Cursor AI Context Extension for EchoZero
 *
 * Automatically injects EchoZero context into AI agent interactions.
 */

const vscode = require('vscode');
const { execSync } = require('child_process');
const path = require('path');

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('EchoZero AI Context Extension activated');

    // Register AI chat interceptor
    const aiChatInterceptor = vscode.commands.registerCommand('echozero.aiContext', async () => {
        // This would be called by Cursor's AI chat system
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            return;
        }

        const document = activeEditor.document;
        const selection = activeEditor.selection;
        const selectedText = document.getText(selection);

        if (selectedText) {
            // Get context for selected text
            const context = await getEchoZeroContext(selectedText);
            if (context) {
                // Insert context as comment above selection
                const contextComment = formatContextAsComment(context);
                const insertPosition = new vscode.Position(selection.start.line, 0);

                await activeEditor.edit(editBuilder => {
                    editBuilder.insert(insertPosition, contextComment + '\n');
                });
            }
        }
    });

    // Auto-inject context on AI chat messages (when available)
    const aiMessageInterceptor = vscode.workspace.onDidChangeTextDocument(async (event) => {
        // Check if this is an AI chat document
        if (event.document.uri.scheme === 'ai-chat' || event.document.fileName.includes('ai-chat')) {
            for (const change of event.contentChanges) {
                const changedText = change.text.trim();

                // Check if this looks like a user message to AI
                if (changedText.length > 10 && !changedText.startsWith('Assistant:') && !changedText.startsWith('AI:')) {
                    // This might be a user message - inject context if it's EchoZero-related
                    const context = await getEchoZeroContext(changedText);
                    if (context) {
                        // Show context in a notification or sidebar
                        showContextNotification(context);
                    }
                }
            }
        }
    });

    context.subscriptions.push(aiChatInterceptor);
    context.subscriptions.push(aiMessageInterceptor);

    // Register context provider for AI agents
    vscode.commands.registerCommand('echozero.getContext', async (userInput) => {
        const context = await getEchoZeroContext(userInput);
        return context;
    });
}

/**
 * Get EchoZero context for user input
 * @param {string} userInput
 * @returns {Promise<Object|null>}
 */
async function getEchoZeroContext(userInput) {
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return null;
        }

        const scriptPath = path.join(workspaceFolder.uri.fsPath, '.cursor', 'ai_context_hook.py');
        const pythonCmd = `python3 "${scriptPath}" "${userInput.replace(/"/g, '\\"')}" cursor_ai`;

        const output = execSync(pythonCmd, {
            cwd: workspaceFolder.uri.fsPath,
            encoding: 'utf8',
            timeout: 10000
        });

        // The script outputs the enhanced prompt
        return {
            enhancedPrompt: output.trim(),
            originalInput: userInput,
            contextProvided: true
        };

    } catch (error) {
        console.error('EchoZero context error:', error);
        return null;
    }
}

/**
 * Format context as code comment
 * @param {Object} context
 * @returns {string}
 */
function formatContextAsComment(context) {
    const lines = [
        '/*',
        ' * 🎯 EchoZero Context Provided',
        ' *',
        ' * This context has been automatically injected for your AI agent.',
        ' * The agent should use this to implement following EchoZero patterns.',
        ' */'
    ];

    return lines.join('\n');
}

/**
 * Show context in a notification
 * @param {Object} context
 */
function showContextNotification(context) {
    const message = 'EchoZero context available for your request. Check the enhanced prompt in your AI chat.';

    vscode.window.showInformationMessage(message, 'View Context').then(selection => {
        if (selection === 'View Context') {
            // Show context in output channel or webview
            const outputChannel = vscode.window.createOutputChannel('EchoZero Context');
            outputChannel.show();
            outputChannel.appendLine('=== EchoZero Context ===');
            outputChannel.appendLine(JSON.stringify(context, null, 2));
        }
    });
}

function deactivate() {
    console.log('EchoZero AI Context Extension deactivated');
}

module.exports = {
    activate,
    deactivate
};

