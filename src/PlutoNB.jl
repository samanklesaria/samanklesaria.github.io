module PlutoNB

__precompile__()

using Pluto: ServerSession, SessionActions, frontmatter
using JSON
using Base64
using StringEncodings

function process_markdown(code)
    replace(code, r"^md\"+" => "",
        r"\"+$" => "",
        r"```math\n((.(?!```))+)\n```"s => s"$$\n\1\n$$")
end

function process_cell(cell)
    if occursin(r"^md", cell.code)
        Dict("metadata" => Dict(),
            "cell_type" => "markdown", "source" => [process_markdown(cell.code)])
    else
        Dict("metadata" => Dict(), "execution_count" => 1,
            "cell_type" => "code", "source" => [cell.code],
            "outputs" => reduce(vcat, [
                get_execution_result(cell.output.body, cell.output.mime),
                get_logs(cell.logs)]))
    end
end

get_logs(logs) = [Dict(
    "output_type" => "stream",
    "name" => "stdout",
    "text" => [l["msg"][1] for l in logs])]

const cell_out = Dict(
    "execution_count" => 1, "metadata" => Dict(), "output_type" => "execute_result")

mimestr(mime::MIME{T}) where {T} = T

get_execution_result(body::Vector, mime::MIME"image/svg+xml") = [
    merge(cell_out, Dict("data" => Dict(mimestr(mime) => [decode(body, "UTF-8")])))]

get_execution_result(body::Vector, mime::MIME"image/png") = [
    merge(cell_out, Dict("data" => Dict(mimestr(mime) => base64encode(body))))]

get_execution_result(body::String, mime) = isempty(body) ? [] : [
    merge(cell_out, Dict("data" => Dict(mimestr(mime) => [body])))]

function get_execution_result(body, mime::MIME"application/vnd.pluto.tree+object")
    reduce(vcat, [get_execution_result(b[2]...) for b in body[:elements]])
end


function jl2nb(in_path::AbstractString, out_path::AbstractString)
    SESSION = ServerSession()
    nb = SessionActions.open(SESSION, in_path; run_async=false)
    order = nb.cell_order
    nbcells = map(order) do cell_uuid
        process_cell(nb.cells_dict[cell_uuid])
    end

    nb_json = Dict("cells" => nbcells,
        "metadata" => merge(
            frontmatter(nb),
            Dict("language_info" => Dict("name" => "julia"))),
        "nbformat" => 4,
        "nbformat_minor" => 2)
    mkpath(dirname(out_path))
    open(out_path, "w") do f
        JSON.print(f, nb_json, 2)
    end
end

end
