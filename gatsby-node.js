const path = require("path")
exports.createPages = async ({ graphql, actions }) => {
	const { data } = await graphql(`
		query posts {
			allMarkdownRemark {
				nodes {
					frontmatter {
						slug
					}
				}
			}
		}
	`)
	const nodes = data.allMarkdownRemark.nodes
	nodes.forEach(node => {
		actions.createPage({
			path: `/posts/${node.frontmatter.slug}`,
			component: path.resolve("./src/templates/post-details.js"),
			context: { slug: node.frontmatter.slug },
		})
	})
}
// exports.createSchemaCustomization = ({ actions }) => {
// 	const { createTypes } = actions
// 	const typeDefs = `
//         type MarkdownRemarkFrontmatter {
//             title: String
//             category: String
//             slug: String
//             date: Date @dateformat
//             featuredImage: File @fileByRelativePath
//         }
//     `
// 	createTypes(typeDefs)
// }

// exports.onCreateNode = ({ node }) => {
// 	if(node.internal.mediaType=="text/markdown"){
//         createNodeField({
//             node,
//             name: `featuredImage`,
//             value:
//         })
//     }
// }
